import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotnine as pn
from sklearn.cluster import KMeans
from scipy.stats import mannwhitneyu


data = pd.read_csv(
    "/content/CDNOW_master.txt",
    sep="\s+",
    names=["c_id", "date", "quantity", "price"],
)

#  0   c_id      69659 non-null  int64
#  1   date      69659 non-null  int64  -> !
#  2   quantity  69659 non-null  int64
#  3   price     69659 non-null  float64

data = data \
        .assign(date=lambda d: d["date"].astype(str)) \
        .assign(date=lambda d: pd.to_datetime(d["date"])) \
        .dropna()

#  0   c_id      69659 non-null  int64
#  1   date      69659 non-null  datetime64[ns]
#  2   quantity  69659 non-null  int64
#  3   price     69659 non-null  float64

first_purchase = (
    data.sort_values(["date", "c_id"])
    .groupby("c_id")
    .first()
)

first_date = first_purchase["date"].min()
last_date = first_purchase["date"].max()

ids = data["c_id"].unique()
selected_ids = ids[:10]

sub_data = data \
    [data["c_id"].isin(ids_selected)] \
    .groupby(["date", "c_id"]) \
    .sum() \
    .reset_index()

# purchase_patterns.png
ggplot(sub_data) \
+ aes(x = "date", y = "price", group = "c_id") \
+ geom_line() \
+ geom_point() \
+ facet_wrap("c_id", scales='free') \
+ theme(axis_text_y=element_blank()) \
+ scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y"
)

n_days = 90
max_date = data["date"].max()
cutdate = max_date - pd.to_timedelta(n_days, unit="d")

tmp_before_cut = data[data["date"] <= cutdate]
tmp_after_cut = data[data["date"] > cutdate]

target_df = tmp_after_cut \
    .drop("quantity", axis=1) \
    .groupby("c_id") \
    .sum() \
    .rename({"price": "spend_90_total"}, axis=1) \
    .assign(spend_90_flag=1)

max_date = tmp_before_cut["date"].max()

recency_df = tmp_before_cut \
            [["c_id", "date"]] \
            .groupby("c_id") \
            .apply(
                lambda d: (d["date"].max() - max_date) / pd.to_timedelta(1, "day")
            ) \
            .to_frame() \
            .set_axis(["recency"], axis=1)

freq_df = tmp_before_cut \
    [["c_id", "date"]] \
    .groupby("c_id") \
    .count() \
    .set_axis(["frequency"], axis=1)

price_df = tmp_before_cut \
    [["c_id", "price"]] \
    .groupby("c_id") \
    .aggregate({
        "price": ["sum", "mean"],
    }) \
    .set_axis(["price_sum", "price_mean"], axis=1)

features = pd.concat(
    [recency_df, freq_df, price_df], axis=1
) \
    .merge(
        target_df,
        left_index=True,
        right_index=True,
        how="left"
    ) \
    .fillna(0)

kmeans = KMeans(n_clusters=3)
features['segment'] = kmeans.fit_predict(features[['recency', 'frequency', 'price_sum']])

# customer_segments_recency_frequency.png
ggplot(features, aes(x='recency', y='frequency')) \
 + geom_point() \
 + facet_wrap("segment") \
 + theme_minimal())

# Mann–Whitney U test

group_1 = features[features['spend_90_flag'] == 1]['price_mean']
group_0 = features[features['spend_90_flag'] == 0]['price_mean']

statistic, p_value = mannwhitneyu(group_1, group_0)

# الان ممكن أستخدام تعلم الالة لتوقع هل العميل سوف يدفع خلال ال 90 يوم القادم أم لا أو توقع كم سيدفع ؟
