import argparse
import os
import optuna
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_visualizations(
    study: optuna.Study,
    filename_prefix: str,
    output_dir: str,
    file_format: str = "html",
):
    """
    Generates and saves a suite of Optuna visualizations for a given study.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    visualizations = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        "slice": optuna.visualization.plot_slice,
        "intermediate_values": optuna.visualization.plot_intermediate_values,
        "timeline": optuna.visualization.plot_timeline,
    }

    for name, plot_function in visualizations.items():
        try:
            fig = plot_function(study)
            # Use the prefix for the output filename
            filename = f"{filename_prefix}_{name}.{file_format}"
            filepath = os.path.join(output_dir, filename)
            if file_format == "html":
                fig.write_html(filepath)
            else:
                # Requires the 'kaleido' package: pip install kaleido
                fig.write_image(filepath)
            logging.info(f"Successfully saved {name} plot to {filepath}")
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            logging.warning(f"Could not generate plot '{name}': {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while generating plot '{name}': {e}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for an Optuna study from a database file."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the Optuna SQLite database file (e.g., 'path/to/your.db').",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optional: The name of the study to visualize. If not provided, it will be inferred from the DB (assuming one study per DB).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the visualization files.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="html",
        choices=["html", "png", "svg", "pdf"],
        help="Output format for the plots (default: html).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        logging.error(f"Database file not found at: {args.db_path}")
        return

    # Get the base name of the database file to use as a prefix for plots
    filename_prefix = os.path.splitext(os.path.basename(args.db_path))[0]

    # Load the study from the database file
    storage_name = f"sqlite:///{args.db_path}"
    study_name = args.study_name

    try:
        # If study_name is not provided, try to infer it
        if not study_name:
            summaries = optuna.get_all_study_summaries(storage=storage_name)
            if len(summaries) == 1:
                study_name = summaries[0].study_name
                logging.info(f"Inferred study name: '{study_name}'")
            elif len(summaries) == 0:
                logging.error(f"No studies found in the database '{args.db_path}'.")
                return
            else:
                logging.error(
                    f"Multiple studies found in '{args.db_path}'. Please specify one using --study_name."
                )
                logging.info("Available studies are:")
                for s in summaries:
                    logging.info(f"  - {s.study_name}")
                return

        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logging.info(
            f"Successfully loaded study '{study_name}' with {len(study.trials)} trials."
        )
    except KeyError:
        logging.error(
            f"Study '{study_name}' not found in the database '{args.db_path}'."
        )
        return
    except Exception as e:
        logging.error(f"Failed to load study: {e}")
        return

    # Generate and save the plots
    generate_visualizations(study, filename_prefix, args.output_dir, args.format)


if __name__ == "__main__":
    main()
