def replace_slash_with_space(csv_path, column_name, output_path=None):
    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        print(f"Column '{column_name}' not found.")
        return

    # Replace all `/` with space in the target column
    df[column_name] = df[column_name].astype(str).str.replace('/', ' ', regex=False)

    print(f"Replaced slashes in column '{column_name}'.")

    # Save to new file or overwrite original
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned CSV to: {output_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"Overwritten original CSV: {csv_path}")


replace_slash_with_space("music_info_cleaned.csv", "artist", "music_info_cleaned_fixed.csv")
replace_slash_with_space("music_info_cleaned.csv", "name", "music_info_cleaned_fixed.csv")
