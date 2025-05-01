# Databricks

## Databricks Asset Bundles

Databricks Asset Bundles are an IaC approach to managing Databricks projects.

You can create them via `databricks bundle init`. This creates:

- `databricks.yml`: name, defintion and settings about workspace
- `resources/<project-name>_job.yml`: job's settings and a default notebook task.
- `src/notebook.ipynb`: sample notebook (can ignore)

Bundles can be validated with `databricks bundle validate`

Bundles can be deployed with `databricks bundle deploy -t dev`, and it should be deployed to:
- Users -> <your-username> -> .bundle -> <project-name> -> dev -> files -> src -> notebook.ipynb

You can then run the deployed project with `databricks bundle run -t dev <project-name>_job`.

To clean up, run `databricks bundle destroy -t dev`

Find more information on configuring bundles [here](https://docs.databricks.com/aws/en/dev-tools/bundles/settings)
