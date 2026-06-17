# Submission Artifact Checksums

Use these checksums to confirm that the uploaded files match the validated repository version.

## Manuscript PDF

| File | Size bytes | SHA256 |
|---|---:|---|
| `manuscript_jae/main.pdf` | 745239 | `A45945DD8A8168E4AAFDDC1C5CA0415C243CC32C353FB94542C9E9C5693A8CA6` |

## Verification Command

On Windows PowerShell:

```powershell
Get-FileHash manuscript_jae\main.pdf -Algorithm SHA256
```

The GitHub Actions workflow also validates the manuscript package on each push to `main`.
