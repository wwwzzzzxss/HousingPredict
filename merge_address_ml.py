import pandas as pd


ADDRESS_FILE = "Address_Finish.csv"
ML_FILE = "ML_Ready_Housing_Data.csv"
OUTPUT_FILE = "Merged_Housing_Data_With_Coordinates.csv"


def read_csv_auto_encoding(path):
    for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode {path}. Please check the file encoding.")


def main():
    address_df = read_csv_auto_encoding(ADDRESS_FILE)
    ml_df = read_csv_auto_encoding(ML_FILE)

    print(f"Address rows: {len(address_df)}")
    print(f"ML rows: {len(ml_df)}")

    address_columns = [
        "id",
        "Address",
        "Response_Address",
        "Response_X",
        "Response_Y",
    ]
    address_df = address_df[address_columns].copy()

    # There is no shared key in ML_Ready_Housing_Data.csv, so merge by row order.
    address_df = address_df.reset_index(drop=True)
    ml_df = ml_df.reset_index(drop=True)

    merged_df = pd.concat([ml_df, address_df], axis=1)
    before_drop = len(merged_df)
    merged_df = merged_df[
        merged_df["Response_X"].notna() & merged_df["Response_Y"].notna()
    ].copy()

    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Done. Output saved to: {OUTPUT_FILE}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Rows removed because coordinates were missing: {before_drop - len(merged_df)}")
    print(f"Merged columns: {len(merged_df.columns)}")


if __name__ == "__main__":
    main()
