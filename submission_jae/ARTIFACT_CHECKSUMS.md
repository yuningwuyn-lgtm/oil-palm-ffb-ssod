# Submission Artifact Checksums

Use these checksums to confirm that the uploaded files match the validated repository version.

## Manuscript PDF

| File | Size bytes | SHA256 |
|---|---:|---|
| `manuscript_jae/main.pdf` | 476812 | `7698AB84B8C657B3811ADF7B9F30BE343A231FC5484F0C0CFA82ACBE7C4654E5` |

## Verification Command

On Windows PowerShell:

```powershell
Get-FileHash manuscript_jae\main.pdf -Algorithm SHA256
```

The GitHub Actions workflow also validates the manuscript package on each push to `main`.
