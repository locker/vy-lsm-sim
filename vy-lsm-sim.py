#! /usr/bin/env python3

import time
import argparse
import matplotlib.pyplot as plot
from heapq import heappush, heappop

parser = argparse.ArgumentParser(
    description=('Simulate Tarantool Vinyl LSM tree performance in case of '
                 'a write-only workload'),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--uniq-key-count', type=int, default=10000000,
                    help='Number of unique keys updated by the workload')
parser.add_argument('--fanout', type=int, default=100,
                    help=('Ratio of the unique key count to the number of '
                          'keys produced by a single memory dump'))
parser.add_argument('--range-count', type=int, default=100,
                    help='Number of key ranges in the LSM tree')
parser.add_argument('--run-size-ratio', type=float, default=3.5,
                    help='Ratio between sizes of adjacent LSM tree levels')
parser.add_argument('--run-count-per-level', type=int, default=2,
                    help='Max number of runs per LSM tree level')
parser.add_argument('--compaction-threads', type=int, default=4,
                    help='Number of threads performing compaction')
parser.add_argument('--compaction-rate', type=float, default=2,
                    help='Ratio of compaction rate to dump rate')
parser.add_argument('--dump-count', type=int, default=1000,
                    help='Number of memory dumps to simulate')
parser.add_argument('--resolution', type=float, default=10,
                    help='Number of times stats are taken between dumps')
parser.add_argument('--read-ampl-pct', type=float, default=0.9,
                    help=('Percentile of {run count: range_count} histogram '
                          'to use for calculating read amplification'))
args = parser.parse_args()

print('Arguments:')
for arg in sorted(vars(args)):
    print('  %s = %s' % (arg, getattr(args, arg)))
print()


class Stat:
    def __init__(self):
        # Number of key-value pairs stored in the LSM tree.
        self.total_size = 0
        # Number of key-value pairs at the largest LSM tree level.
        self.last_level_size = 0
        # Number of completed memory dumps.
        self.dump_count = 0
        # Number of key-value pairs dumped to disk.
        self.dump_out = 0
        # Number of key-value pairs read by compaction.
        self.compaction_in = 0
        # Number of key-value pairs written by compaction.
        self.compaction_out = 0
        # Number of key-value pairs awaiting compaction.
        self.compaction_queue_size = 0
        # Histogram: number of runs => number of ranges.
        self.run_histogram = []

    def account_range(self, range_):
        while range_.run_count >= len(self.run_histogram):
            self.run_histogram.append(0)
        self.run_histogram[range_.run_count] += 1

    def unaccount_range(self, range_):
        assert(self.run_histogram[range_.run_count] > 0)
        self.run_histogram[range_.run_count] -= 1

    @property
    def run_count_max(self):
        for i in reversed(range(len(self.run_histogram))):
            if self.run_histogram[i] > 0:
                return i
        return 0

    @property
    def read_ampl(self):
        range_count = 0
        total_range_count = sum(self.run_histogram)
        for i in range(len(self.run_histogram)):
            range_count += self.run_histogram[i]
            if range_count >= total_range_count * args.read_ampl_pct:
                return i
        return 0

    @property
    def write_ampl(self):
        return 1 + self.compaction_out / (self.dump_out + 1)

    @property
    def space_ampl(self):
        return self.total_size / (self.last_level_size + 1)

    @property
    def compaction_queue(self):
        return self.compaction_queue_size / (self.total_size + 1)

    @property
    def compaction_ratio(self):
        return self.compaction_in / (self.compaction_out + 1)


# Represents a sorted set of key-value pairs on disk.
# All keys stored in a run are unique.
class Run:
    def __init__(self, size):
        # Number of keys stored in the run.
        self.size = size

    # Create a new run by compacting compacted_runs.
    #
    # Let k_i be the number of keys in input run i, L be the total
    # number of input runs, N be the total number of unique keys.
    # Since all keys in a run are distinct, k_i <= N. Let us define
    # I_j as
    #
    #   1 if key j is present in the output run,
    #   0 otherwise
    #
    # for each key 1 <= j <= N.
    #
    # Then the expected number of keys in the output run equals
    #
    #   EXPECTED(SUM_j I_j) = SUM_j EXPECTED(I_j)
    #     = SUM_j PROBABILITY(I_j == 1)
    #     = SUM_j (1 - PROBABILITY(I_j == 0))
    #
    # Probability that a particular key is not present in run i
    # equals
    #
    #   (N-1):choose:k_i / N:choose:k_i = (N - k_i) / N
    #
    # Hence
    #
    #   PROBABILITY(I_j == 0) = MULT_i [(N - k_i) / N]
    #
    # and for the expected number of output keys we have
    #
    #   SUM_j (1 - MULT_i [(N - k_i) / N])
    #     = N * (1 - MULT_i [(N - k_i) / N])
    #
    @classmethod
    def from_compacted_runs(cls, uniq_key_count, compacted_runs):
        mult = 1
        for run in compacted_runs:
            mult *= 1 - run.size / uniq_key_count
        return cls(int(uniq_key_count * (1 - mult)))


# Represents a key range in the simulated LSM tree.
# Different key ranges do not interleave.
class Range:
    def __init__(self, stat):
        # Global statistics.
        self.stat = stat
        # Runs sorted by age: oldest run comes first.
        self.runs = []
        # Number of dumps since the last major compaction.
        self.dumps_since_compaction = 0
        # Number of runs that need to be compacted.
        self.compaction_prio = 0
        # Total size of runs that need to be compacted.
        self.compaction_queue_size = 0
        # If compaction is in progress, slice of compacted runs.
        self.compaction_slice = None

        self.stat.account_range(self)

    # Simulate memory dump.
    def dump(self):
        size = int(args.uniq_key_count / args.fanout / args.range_count)
        self.stat.dump_out += size
        self.stat.total_size += size
        if not self.runs:
            self.stat.last_level_size += size
        self.stat.unaccount_range(self)
        self.dumps_since_compaction += 1
        self.runs.append(Run(size))
        self.stat.account_range(self)
        self.update_compaction_prio()

    # Number of runs in this range.
    @property
    def run_count(self):
        return len(self.runs)

    # True if this range is currently being compacted.
    @property
    def in_compaction(self):
        return self.compaction_slice is not None

    # Start compaction of @run_count newest runs.
    # Returns the sum size of compacted runs.
    def start_compaction(self, run_count):
        assert(not self.in_compaction)
        assert(run_count > 1 and run_count <= self.run_count)
        self.compaction_slice = slice(self.run_count - run_count,
                                      self.run_count)
        if self.compaction_slice.start == 0:  # major compaction
            self.dumps_since_compaction = 0
        return sum(run.size for run in self.runs[self.compaction_slice])

    # Complete compaction started with start_compaction().
    # It replaces compacted runs with the resulting run.
    def complete_compaction(self):
        assert(self.in_compaction)
        compacted_runs = self.runs[self.compaction_slice]
        uniq_key_count = int(args.uniq_key_count / args.range_count)
        new_run = Run.from_compacted_runs(uniq_key_count, compacted_runs)

        input_size = sum(run.size for run in compacted_runs)
        self.stat.compaction_in += input_size
        self.stat.compaction_out += new_run.size
        self.stat.total_size -= input_size
        self.stat.total_size += new_run.size
        if self.compaction_slice.start == 0:  # major compaction
            self.stat.last_level_size -= self.runs[0].size
            self.stat.last_level_size += new_run.size

        self.stat.unaccount_range(self)
        self.runs[self.compaction_slice] = [new_run]
        self.stat.account_range(self)
        self.compaction_slice = None

        self.update_compaction_prio()

    def update_compaction_prio(self):
        # Total number of checked runs.
        total_run_count = 0
        # Total size of checked runs.
        total_size = 0
        # Estimated size of the output run, if compaction is scheduled.
        est_new_run_size = 0
        # Number of runs at the current level.
        level_run_count = 0
        # The target (perfect) size of a run at the current level.
        # For the first level, it's the size of the newest run.
        # For lower levels it's computed as first level run size
        # times run_size_ratio.
        target_run_size = 0

        self.stat.compaction_queue_size -= self.compaction_queue_size
        self.compaction_prio = 0
        self.compaction_queue_size = 0

        for run in reversed(self.runs):
            # The size of the first level is defined by
            # the size of the most recent run.
            if target_run_size == 0:
                target_run_size = run.size

            level_run_count += 1
            total_run_count += 1
            total_size += run.size

            while run.size > target_run_size:
                # The run size exceeds the threshold set for the
                # current level.  Move this run down to a lower
                # level. Switch the current level and reset the
                # level run count.
                level_run_count = 1

                # If we have already scheduled a compaction of an
                # upper level, and estimated compacted run will
                # end up at this level, include the new run into
                # this level right away to avoid a cascading
                # compaction.
                if est_new_run_size > target_run_size:
                    level_run_count += 1

                # Calculate the target run size for this level.
                target_run_size *= args.run_size_ratio

                # Keep pushing the run down until we find an
                # appropriate level for it.

            if level_run_count > args.run_count_per_level:
                # The number of runs at the current level exceeds
                # the configured maximum. Arrange for compaction.
                # We compact all runs at this level and upper
                # levels.
                est_new_run_size = total_size
                self.compaction_prio = total_run_count
                self.compaction_queue_size = total_size

        # Never store more than one run on the last level to keep
        # space amplification low.
        if level_run_count > 1:
            self.compaction_prio = total_run_count
            self.compaction_queue_size = total_size

        self.stat.compaction_queue_size += self.compaction_queue_size


class Event:
    def __init__(self, time, action):
        self.time = time
        self.action = action

    def __lt__(self, other):
        return self.time < other.time


class Timeline:
    def __init__(self):
        self.now = 0
        self.events = []

    def add_event(self, timeout, action):
        heappush(self.events, Event(self.now + timeout, action))

    def process_event(self):
        ev = heappop(self.events)
        self.now = ev.time
        ev.action()


# Responsible for scheduling background compaction.
class Scheduler:
    def __init__(self, stat, timeline, ranges):
        # Global statistics.
        self.stat = stat
        # Timeline object for scheduling events.
        self.timeline = timeline
        # Ranges to schedule compaction for.
        self.ranges = ranges
        # Number of compaction tasks in progress.
        self.compaction_tasks = 0

        self.schedule_dump()

    def schedule_dump(self):
        self.timeline.add_event(1, self.complete_dump)

    def complete_dump(self):
        for it in self.ranges:
            it.dump()
        self.stat.dump_count += 1
        self.schedule_dump()
        self.schedule_compaction()

    def pick_range_for_compaction(self):
        choice = None
        for it in self.ranges:
            if (it.compaction_prio > 1 and not it.in_compaction and
                    (choice is None or
                     it.compaction_prio > choice.compaction_prio)):
                choice = it
        if choice is not None:
            return choice, choice.compaction_prio
        return None, 0

    def schedule_range_compaction(self, range_, run_count):
        assert(self.compaction_tasks < args.compaction_threads)
        self.compaction_tasks += 1
        dump_size = args.uniq_key_count / args.fanout
        compaction_size = range_.start_compaction(run_count)
        compaction_rate = dump_size * args.compaction_rate
        self.timeline.add_event(compaction_size / compaction_rate,
                                lambda: self.complete_range_compaction(range_))

    def complete_range_compaction(self, range_):
        assert(self.compaction_tasks > 0)
        self.compaction_tasks -= 1
        range_.complete_compaction()
        self.schedule_compaction()

    def schedule_compaction(self):
        while self.compaction_tasks < args.compaction_threads:
            range_, run_count = self.pick_range_for_compaction()
            if range_ is None:
                break
            self.schedule_range_compaction(range_, run_count)


class Simulator:
    def __init__(self):
        self.stat = Stat()
        self.timeline = Timeline()
        self.ranges = [Range(self.stat) for _ in range(args.range_count)]
        self.scheduler = Scheduler(self.stat, self.timeline, self.ranges)

        self.stat_funcs = ('read_ampl', 'write_ampl',
                           'space_ampl', 'compaction_queue')
        self.stat_x = []
        self.stat_y = {}
        for f in self.stat_funcs:
            self.stat_y[f] = []

    def run(self):
        print('Running simulation...')
        start = time.time()
        self.timeline.add_event(0, self.report)
        while self.stat.dump_count < args.dump_count:
            self.timeline.process_event()
        stop = time.time()
        print('Simulation completed in %0.2f seconds' % (stop - start))
        print()

        print('Plotting data...')
        assert(len(self.stat_funcs) == 4)
        for i in range(len(self.stat_funcs)):
            f = self.stat_funcs[i]
            plot.subplot(2, 2, i + 1)
            plot.plot(self.stat_x, self.stat_y[f])
            plot.xlabel('dump-count')
            plot.ylabel(f)
            plot.grid()
        plot.show()

    def report(self):
        self.stat_x.append(self.timeline.now)
        for f in self.stat_funcs:
            self.stat_y[f].append(getattr(self.stat, f))
        self.timeline.add_event(1 / args.resolution, self.report)


Simulator().run()
