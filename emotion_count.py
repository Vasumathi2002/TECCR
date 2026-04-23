import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = "dataset.csv"

PRIMARY_COL = "primary_emotion"
SECONDARY_COL = "secondary_emotion"

# ==============================
# LOAD DATASET
# ==============================
print("📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print("✅ Dataset loaded successfully")
print("Columns found:", df.columns.tolist())

# ==============================
# BASIC COUNTS
# ==============================
total_rows = len(df)

primary_not_null = df[PRIMARY_COL].notna().sum()
secondary_not_null = df[SECONDARY_COL].notna().sum()

print("\n==============================")
print("📊 BASIC SUMMARY")
print("==============================")
print(f"Total Records           : {total_rows}")
print(f"Primary Emotion Entries : {primary_not_null}")
print(f"Secondary Emotion Entries: {secondary_not_null}")

# ==============================
# PRIMARY EMOTION COUNTS
# ==============================
primary_counts = df[PRIMARY_COL].value_counts()

print("\n==============================")
print("🎭 PRIMARY EMOTION COUNTS")
print("==============================")
for emotion, count in primary_counts.items():
    print(f"{emotion} : {count}")

# ==============================
# SECONDARY EMOTION COUNTS
# (Ignoring 'None' values)
# ==============================
secondary_filtered = df[SECONDARY_COL][
    (df[SECONDARY_COL].notna()) &
    (df[SECONDARY_COL].str.lower() != "none")
]

secondary_counts = secondary_filtered.value_counts()

print("\n==============================")
print("🎭 SECONDARY EMOTION COUNTS")
print("==============================")
for emotion, count in secondary_counts.items():
    print(f"{emotion} : {count}")

# ==============================
# COMBINED EMOTION ANALYSIS
# ==============================
combined_emotions = pd.concat([
    df[PRIMARY_COL],
    secondary_filtered
])

combined_counts = combined_emotions.value_counts()

print("\n==============================")
print("🔥 COMBINED EMOTION COUNTS")
print("==============================")
for emotion, count in combined_counts.items():
    print(f"{emotion} : {count}")

summary_df = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Primary Emotion Entries",
        "Secondary Emotion Entries"
    ],
    "Count": [
        total_rows,
        primary_not_null,
        secondary_not_null
    ]
})

print("\n✅ Emotion analysis completed.")
