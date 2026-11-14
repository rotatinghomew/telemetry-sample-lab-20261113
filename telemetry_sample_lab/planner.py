"""Command-line interface for design, sample, estimate and replay validation.

Entry point: ``python3 -m telemetry_sample_lab.planner``.

Subcommands
-----------
``design``   ``--input`` design document, ``--output`` saved plan.
             Reads only design-time keys; ``replay_population`` is never read
             and the design digest covers design keys only, so a saved plan
             can be audited later without the future results (or with them
             altered).
``sample``   ``--input`` delivery document with population rows (the fixture
             key ``replay_population``, or ``population_rows``), ``--plan``
             saved plan, ``--output`` executed sample with per-row inclusion
             probabilities.  Optional ``--seed`` to draw with another seed.
``estimate`` ``--input`` executed sample, ``--plan`` saved plan,
             ``--output`` design-based estimates and intervals.
``replay``   ``--input`` document with future rows and ``--plan`` saved plan,
             ``--output`` repeated-replay comparison with the uniform-1%
             baseline.  Optional ``--seeds N`` (default 200).

``--input`` and ``--output`` are required by every command; ``--plan`` is
required by everything except ``design``.
"""

import argparse
import sys

from .design import design, verify_plan
from .estimate import estimate
from .execute import sample
from .io import read_json, write_json
from .replay import replay


def _parser():
    parser = argparse.ArgumentParser(
        prog='telemetry_sample_lab.planner',
        description='多指标分层设备簇抽样：设计、执行、估计与回放检查。')
    parser.add_argument('command', choices=['design', 'sample', 'estimate', 'replay'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=200,
                        help='replay 子命令使用的重放次数（种子 1..N）')
    parser.add_argument('--min-devices', type=int, default=2,
                        help='design 时每个分层至少保留的设备数')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        document = read_json(args.input)
        if args.command == 'design':
            plan = design(document, {'min_devices': args.min_devices})
            write_json(args.output, plan)
            print('计划已保存：' + args.output)
            return 0
        if args.plan is None:
            parser.error(args.command + ' 需要 --plan')
        plan = read_json(args.plan)
        verify_plan(plan)
        if args.command == 'sample':
            execution = sample(document, plan, seed=args.seed)
            write_json(args.output, execution)
            print('样本已保存：' + args.output)
            return 0
        if args.command == 'estimate':
            result = estimate(document, plan)
            write_json(args.output, result)
            print('估计已保存：' + args.output)
            return 0
        if args.command == 'replay':
            if args.seeds <= 0:
                parser.error('--seeds 必须是正整数')
            result = replay(plan, document, range(1, args.seeds + 1))
            write_json(args.output, result)
            print('回放检查已保存：' + args.output)
            return 0
        parser.error('未知子命令：' + args.command)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    except MemoryError:
        print('处理失败：内存不足，请减小 --seeds', file=sys.stderr)
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
