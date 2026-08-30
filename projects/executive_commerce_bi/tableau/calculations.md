# Tableau Calculation Pack

The tidy Tableau export contains `section`, `dimension`, `metric`, `value`, `secondary_value`, and `sort_order`.

## Selected Metric Value

```tableau
SUM([value])
```

Use together with a `[metric]` filter so each worksheet has one semantic value.

## KPI Label

```tableau
IF [section] = "Executive KPI" THEN [dimension] END
```

## Delivery Review Gap

```tableau
WINDOW_MAX(IF [dimension] = "on_time" AND [metric] = "Average Review" THEN [value] END)
-
WINDOW_MAX(IF [dimension] = "late" AND [metric] = "Average Review" THEN [value] END)
```

## Category Rank Filter

```tableau
[section] = "Category" AND [metric] = "Category GMV" AND [sort_order] <= 10
```

## Display rules

- currencies use `R$ #,##0.0K` / `R$ #,##0.0M` depending on scale
- percentages show two decimal places
- review scores show two decimals on a 1–5 scale
- category views sort by `sort_order`, not alphabetically
- never aggregate percentages across unrelated dimensions
