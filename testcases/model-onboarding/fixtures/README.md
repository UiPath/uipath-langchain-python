# Test fixtures

Both files are generated, committed, and read from disk by
`src/agents/file_processing/agent.py`. They are deliberately tiny (608 B and
1.9 KB).

| File | Content | Question asked | Expected |
|---|---|---|---|
| `document.pdf` | One page: `Verification code: PDF-CODE-74915` | "What is the verification code written in this document?" | `PDF-CODE-74915` |
| `shape.png` | 512×512, purple square (`#800080`) on white | "What colour is the large shape in the centre?" | `purple` |

## Why these, and not files from the web

The answers are **unguessable**, which is the only reason the assertion proves
anything. A model that never opened the file cannot produce a random code, and
has no prior for an arbitrary colour.

The previous fixtures were borrowed URLs and both were guessable:

- `dog.jpg` was asked "what animal is in this image?" — "dog" is the single
  most likely answer to that question with no image at all, and the file name
  appears in the prompt. A model that skipped the file scored correct.
- `dummy.pdf` contained the literal text "Dummy PDF file", which reads as a
  placeholder. The model kept commenting on whether the content was real
  instead of reporting it, failing in both directions across runs.

Compensating for guessable answers previously required a refusal-phrase
blocklist and a position-in-the-answer heuristic in `_matches`. Both were
brittle, and neither was needed once the fixtures changed: a plain token match
now suffices.

Local rather than remote also means the expected answer is a property of bytes
in this repo — a third-party host cannot silently change the file and break or,
worse, quietly weaken the test.

## Regenerating

`document.pdf` is written uncompressed on purpose, so the code stays greppable
in the raw bytes and can be checked without a PDF library:

```bash
grep -c 'PDF-CODE-74915' fixtures/document.pdf   # 1
```

If you change a fixture, update `expected` in `FILE_REGISTRY`
(`src/main.py`) to match. The generator scripts are not committed — these are
static test inputs, and regenerating them is a deliberate act, not part of the
build.
