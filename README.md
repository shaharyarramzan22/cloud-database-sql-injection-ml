# Dataset

Place the original thesis dataset here as `sql_queries.csv`.

Required columns:

- `query`: raw SQL query text
- `label`: `0` benign, `1` malicious

Example:

```csv
query,label
"SELECT name FROM users WHERE id = 10",0
"SELECT name FROM users WHERE id = 10 OR 1=1",1
```

Do not commit confidential production database logs or personally identifiable information.
