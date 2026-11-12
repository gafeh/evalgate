# EvalGate

An evaluation harness for LLM graders. It scores a model's rubric decisions against
human ground truth, measures whether its own judge can be trusted, and refuses to
call a change an improvement unless the statistics support it.

The subject under evaluation here is an AP Computer Science A grader: given a
free-response question, an official-style rubric and a student's Java, it awards
rubric items and writes feedback to the student.

## What it found

Claude Haiku 5.5 grading 60 authored cases across four questions:

| | |
|---|---|
| mean score against gold awards | 0.7975 |
| cases matching the rubric exactly | 76.7% |
| rubric points awarded that weren't earned | 39 |
| rubric points missed that were earned | 7 |

The over-crediting is five times the under-crediting, and almost all of it sits in
one place. Broken out by error mode, the grader scores between 0.51 and 1.00 on
every family of mistake except one:

```
  correct            n=13   mean=1.000  over=0
  off_by_one         n=7    mean=1.000  over=0
  reference_equality n=2    mean=1.000  over=0
  return_value       n=5    mean=0.920  over=0
  traversal_shift    n=2    mean=0.700  over=0
  logic              n=3    mean=0.667  over=0
  syntax             n=8    mean=0.000  over=38
```

Eight cases of code that does not compile, scored 0.000, carrying 38 of the 39
unearned points. The grader reasons about what the student meant and treats syntax
as a cosmetic aside. On two of those cases it awarded full marks and then listed the
missing semicolons as a closing note. On another it wrote "under the rubric as given,
the logic earns full credit" about code that cannot be compiled.

A student reading that feedback is told their broken program is correct.

## Why over-crediting is the metric

Most eval harnesses report accuracy. In grading, the two directions of error are not
interchangeable. Missing a point a student earned gets caught the moment they
complain. Awarding a point they didn't earn goes onto a transcript. So every run
reports the two separately and never nets them off, and the regression gate checks
over-crediting before it looks at the aggregate score. A candidate that scores
better overall while handing out more unearned points is a regression.

## Measuring the measuring instrument

An LLM judge reviews each piece of grader feedback and reports a verdict with a
confidence number. That number is a claim, not a measurement, so the project checks
it: 50 cases were labelled by hand, blind to the judge's verdict, and the judge was
scored against those labels.

```
labelled calibration pairs: 50
distinct cases:             50
human agreement:            0.940
cohen's kappa:              0.796
brier score:                0.0433
expected calibration error: 0.0638
mean judge confidence:      0.911

reliability curve
  bin               n   claimed   actual      gap
  0.4-0.6           2     0.575    0.500   -0.075
  0.6-0.8           1     0.720    0.000   -0.720
  0.8-1.0          47     0.929    0.979   +0.049

agreement by judge verdict
  fail      0.833
  partial   0.500
  pass      0.976
```

Kappa of 0.796 matters more than the raw 0.940, because a judge that always says
"pass" on a mostly-passing set would score high agreement by accident. The judge is
mildly underconfident rather than overconfident, and it is far more reliable when
approving than when rejecting.

Two properties of the design make these numbers mean something.

**Labelling is blind structurally, not by convention.** The object handed to the
labelling interface has no verdict or confidence field on it. Not hidden by the
renderer, absent from the type, and a test asserts it stays that way. If you can see
the judge's answer while labelling, the agreement number you compute afterwards is
circular.

**The calibration sample is drawn before triage, at random.** The obvious design
routes low-confidence results to review. Label only those and you have measured the
judge exactly where it said it was unsure, which tells you nothing about the
high-confidence cases you were about to auto-approve. So there are two populations:
a random sample for measurement, and a triage queue for action. The metrics only
ever use the first.

### What the calibration can't tell you yet

The report refuses to oversell itself. On this data it recommends a review threshold
and then says why not to use it:

```
recommended review threshold: 0.550
  above it: 100.0% of cases at 94.0% accuracy
  below it: 0 pairs at 0.0% accuracy
  ! this cut sends nothing to review, so it is not a working threshold
  ! only 0 labelled pairs fall below the cut, so its low side is unmeasured
```

The judge reports 0.8 or higher on 94% of cases, so a random sample of 50 contains
almost no low-confidence examples and no threshold claim is supported. The fix is
stratified sampling with weights carried through the metrics, and it isn't built yet.
Until it is, the honest answer is that the threshold is unknown.

## Deciding whether a change is real

Two runs over the same cases are compared with a paired bootstrap on the
per-case difference, and the verdict has four states rather than two:

- **regressed** if over-crediting rose significantly, or the score interval sits
  below zero by more than the minimum effect
- **improved** if the interval sits above zero by more than the minimum effect
- **no change** if the whole interval fits inside the minimum effect band, which
  rules out a change worth acting on
- **inconclusive** if the interval straddles zero and is too wide to rule one out

Only the middle two pass. "Inconclusive" failing the gate is the point: it is not the
same as "nothing changed", and a gate that treats them alike waves regressions
through whenever the dataset is small. When a comparison is inconclusive the tool
estimates how many more paired cases would settle it.

The same 30% over-crediting injected into 200 cases is a confident regression, and
into 12 cases is an honest "not enough data, you'd need about 217".

## Tamper-evident results

A finished run is sealed into a hash chain, one link per result, each covering
everything before it. Three distinct forgeries are detectable: editing a result row,
editing a row together with its own fingerprint, and rewriting the chain itself.

```
$ psql -c "update results set overall = 1.0 where overall = 0.0"
UPDATE 9

$ evalgate verify 6b934641-3ae6-4923-b4b8-de0c8383c138
links: 60
root:  ac573b03cd7fba424330e158c1cb3043f8fd767c0a640b4a955dc9b841e190b3
integrity: FAILED
  seq 7:  row: case count-length-as-method no longer matches its own fingerprint
  seq 8:  row: case count-missing-paren no longer matches its own fingerprint
  seq 17: row: case remove-clears-everything no longer matches its own fingerprint
  ...
```

That edit rewrites exactly the eight non-compiling cases to look like full marks,
which is the specific lie someone would tell about this run. All nine rows are named
by case id. The chain also doubles as a recovery record, since each link carries the
score it was sealed with.

## Running it

Python 3.12 and Docker.

```bash
docker compose up -d                 # Postgres on 5433
uv sync
uv run alembic upgrade head

uv run evalgate dataset load datasets/apcs --version 1
RUN=$(uv run evalgate run start --dataset apcs-frq@1 --model claude-haiku-5-5)
uv run evalgate grade "$RUN"         # needs ANTHROPIC_API_KEY, costs about $0.80
uv run evalgate run status "$RUN"
```

Then measure the judge:

```bash
uv run evalgate review enqueue "$RUN" --sample 50
uv run evalgate review label --run-id "$RUN" --labeller you
uv run evalgate calibrate --run-id "$RUN"
```

Then gate a change and seal the record:

```bash
uv run evalgate run compare "$CANDIDATE" "$BASELINE"   # exit 1 on a regression
uv run evalgate seal "$RUN"
uv run evalgate verify "$RUN"
uv run evalgate report "$RUN" --baseline "$BASELINE" --out report.json
```

`evalgate worker` runs the same pipeline against a deterministic stub grader, so the
queue, the gate and the audit chain can all be exercised without spending anything.
`evalgate serve` exposes the same data over HTTP with OpenAPI docs at `/docs`.

## How it's built

Python throughout. FastAPI, Postgres 16, SQLAlchemy 2 in async mode with Alembic
migrations, Pydantic at every boundary, Typer for the CLI, pytest, ruff and mypy in
strict mode. 272 tests.

Postgres is also the work queue. Workers claim batches with
`SELECT ... FOR UPDATE SKIP LOCKED`, heartbeat while they work, and anything a dead
worker was holding is returned to the queue by whoever notices. Delivery is
at-least-once and the unique constraint on `(run_id, case_id)` is what makes that
safe, so a retry after a crash bounces off the database rather than double-counting.
Kill a worker mid-run and every case is still graded exactly once.

```
src/evalgate/
  core/        pure logic: scorers, statistics, calibration, the hash chain
  db/          models, queue, repositories, review
  judge/       API client, response cache, cost ledger, the feedback judge
  runner/      grading loop and workers
  api/         FastAPI routers
  cli/         the evalgate command
datasets/apcs/ the authored case corpus
```

`core/` imports no database driver, no web framework and no HTTP client, and a test
walks its imports to keep it that way. Most of the suite runs in milliseconds with
no database at all.

Model responses are cached by a hash of everything that determines the answer, and
every call is logged to a separate append-only ledger. A rerun of an identical
evaluation reports the same work at zero cost, which is also where reproducibility
comes from: the API no longer accepts sampling parameters, so replaying a cached
response is the only way to get the same answer twice.

## Limitations

There is one labeller. Agreement with a single human is not the same as agreement
with the rubric, and there's no second rater to measure how consistent that human is.

The judge's calibration is unmeasured below 0.8 confidence, for the reason given
above.

The per-pull-request CI gate grades with the stub so it runs free and
deterministically on every change. The real-model gate is a separate workflow run by
hand, because grading 60 cases against a live API on every push costs money.

Model prices are hardcoded in `judge/pricing.py` and need checking against current
rates. A stale constant there makes every cost figure wrong and nothing in the code
can detect it.
