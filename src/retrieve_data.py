#!/usr/bin/env python3

import argparse
import sys
from google.cloud import bigquery


def get_table_metadata(client, project_id, dataset_id=None, target_project=None):
    # Use target_project if specified, otherwise use project_id
    data_project = target_project if target_project else project_id

    if dataset_id:
        query = f"""
        SELECT
            t.table_catalog,
            t.table_schema,
            t.table_name,
            STRING_AGG(c.column_name, ', ' ORDER BY c.ordinal_position) as all_columns
        FROM `{data_project}.{dataset_id}.INFORMATION_SCHEMA.TABLES` t
        LEFT JOIN `{data_project}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS` c
        USING(table_catalog, table_schema, table_name)
        GROUP BY 1,2,3
        ORDER BY t.table_schema, t.table_name
        """
    else:
        # Get all datasets first, then query each one
        datasets_query = f"""
        SELECT schema_name as dataset_id
        FROM `{data_project}.INFORMATION_SCHEMA.SCHEMATA`
        WHERE schema_name NOT IN ('INFORMATION_SCHEMA')
        """

        try:
            datasets_result = client.query(datasets_query)
            datasets = [row.dataset_id for row in datasets_result]
        except Exception as e:
            print(f"Error getting datasets: {e}", file=sys.stderr)
            return []

        all_tables = []
        for dataset in datasets:
            try:
                dataset_query = f"""
                SELECT
                    t.table_catalog,
                    t.table_schema,
                    t.table_name,
                    STRING_AGG(
                    c.column_name,
                    ', '
                    ORDER BY c.ordinal_position
                    ) as all_columns
                FROM `{data_project}.{dataset}.INFORMATION_SCHEMA.TABLES` t
                LEFT JOIN `{data_project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` c
                USING(table_catalog, table_schema, table_name)
                GROUP BY 1,2,3
                ORDER BY t.table_schema, t.table_name
                """
                result = client.query(dataset_query)
                for row in result:
                    all_tables.append(
                        (
                            row.table_catalog,
                            row.table_schema,
                            row.table_name,
                            row.all_columns,
                        )
                    )
            except Exception as e:
                print(
                    f"Warning: Could not query dataset {dataset}: {e}", file=sys.stderr
                )
                continue

        return all_tables

    try:
        result = client.query(query)
        tables = []
        for row in result:
            tables.append(
                (row.table_catalog, row.table_schema, row.table_name, row.all_columns)
            )
        return tables
    except Exception as e:
        print(f"Error executing query: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="""
Retrieve BigQuery table metadata using INFORMATION_SCHEMA
and export it into pipe-separated file named output.csv

The script assumes users have:
  - GCP CLI installed and authenticated
  - BigQuery API enabled
  - Proper project access permission

The current script will work with these prerequisites using the default credentials
from gcloud auth application-default login.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project_id", help="Google Cloud Project ID (your project for running jobs)"
    )
    parser.add_argument("--dataset", help="Specific dataset ID (optional)")
    parser.add_argument(
        "--target-project",
        help="Target project containing the dataset (default: same as project_id)",
    )
    parser.add_argument("--output", "-o", help="Output file (CSV format)")
    parser.add_argument("--limit", type=int, help="Limit number of results")

    args = parser.parse_args()

    try:
        client = bigquery.Client(project=args.project_id)

        target_project = (
            args.target_project if hasattr(args, "target_project") else None
        )
        tables = get_table_metadata(
            client, args.project_id, args.dataset, target_project
        )

        if args.limit:
            tables = tables[: args.limit]

        # Output pipe-separated CSV header
        header = "table_catalog|table_schema|table_name|all_columns"

        output_file = args.output if args.output else "output.csv"

        with open(output_file, "w") as f:
            f.write(header + "\n")
            for table in tables:
                line = "|".join(
                    str(field) if field is not None else "" for field in table
                )
                f.write(line + "\n")
        print(f"Metadata for {len(tables)} tables saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
