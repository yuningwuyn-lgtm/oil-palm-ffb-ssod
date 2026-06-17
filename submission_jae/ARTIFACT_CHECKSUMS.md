# Submission Artifact Checksums

Use these checksums to confirm that the uploaded files match the validated repository version.

## Manuscript PDF

| File | Size bytes | SHA256 |
|---|---:|---|
| `manuscript_jae/main.pdf` | 477933 | `1EF690168D161C5662338204B67F8C847CD7C336E8382DF17A4FF5380E474EF5` |

## Verification Command

On Windows PowerShell:

```powershell
Get-FileHash manuscript_jae\main.pdf -Algorithm SHA256
```

The GitHub Actions workflow also validates the manuscript package on each push to `main`.
