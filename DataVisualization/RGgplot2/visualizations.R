# R Data Visualization with ggplot2
# Professional statistical graphics and publication-ready plots

# Install required packages if needed
# install.packages(c("ggplot2", "dplyr", "tidyr", "scales", "gridExtra"))

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(gridExtra)

# Set theme for all plots
theme_set(theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "gray40"),
    axis.title = element_text(size = 12, face = "bold"),
    legend.position = "top"
  ))

# ========== 1. SCATTER PLOT WITH REGRESSION ==========
scatter_plot <- function() {
  set.seed(123)
  data <- data.frame(
    x = rnorm(100, 50, 15),
    y = rnorm(100, 50, 15),
    category = sample(c("A", "B", "C"), 100, replace = TRUE)
  )

  p <- ggplot(data, aes(x = x, y = y, color = category)) +
    geom_point(size = 3, alpha = 0.6) +
    geom_smooth(method = "lm", se = TRUE, linetype = "dashed") +
    scale_color_manual(values = c("#667eea", "#764ba2", "#f093fb")) +
    labs(
      title = "Scatter Plot with Regression Lines",
      subtitle = "Correlation analysis by category",
      x = "Variable X",
      y = "Variable Y",
      color = "Category"
    )

  print(p)
  ggsave("scatter_plot.png", width = 10, height = 6, dpi = 300)
  return(p)
}

# ========== 2. BOX PLOT ==========
box_plot <- function() {
  data <- data.frame(
    category = rep(c("Product A", "Product B", "Product C", "Product D"), each = 50),
    value = c(rnorm(50, 100, 15), rnorm(50, 120, 20),
              rnorm(50, 90, 12), rnorm(50, 110, 18))
  )

  p <- ggplot(data, aes(x = category, y = value, fill = category)) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
    scale_fill_manual(values = c("#667eea", "#764ba2", "#f093fb", "#4facfe")) +
    labs(
      title = "Distribution Comparison with Box Plots",
      subtitle = "Diamond indicates mean value",
      x = "Product Category",
      y = "Value"
    ) +
    theme(legend.position = "none")

  print(p)
  ggsave("box_plot.png", width = 10, height = 6, dpi = 300)
  return(p)
}

# ========== 3. TIME SERIES LINE PLOT ==========
time_series_plot <- function() {
  dates <- seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "day")
  data <- data.frame(
    date = dates,
    sales = cumsum(rnorm(length(dates), 100, 50)) + 5000,
    revenue = cumsum(rnorm(length(dates), 150, 60)) + 8000
  )

  data_long <- data %>%
    pivot_longer(cols = c(sales, revenue), names_to = "metric", values_to = "value")

  p <- ggplot(data_long, aes(x = date, y = value, color = metric)) +
    geom_line(size = 1.2) +
    scale_color_manual(
      values = c("sales" = "#667eea", "revenue" = "#764ba2"),
      labels = c("Sales", "Revenue")
    ) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b") +
    scale_y_continuous(labels = comma) +
    labs(
      title = "Time Series Analysis",
      subtitle = "Sales and Revenue trends throughout 2023",
      x = "Date",
      y = "Value ($)",
      color = "Metric"
    )

  print(p)
  ggsave("time_series.png", width = 12, height = 6, dpi = 300)
  return(p)
}

# ========== 4. BAR CHART WITH ERROR BARS ==========
bar_chart <- function() {
  data <- data.frame(
    category = c("Q1", "Q2", "Q3", "Q4"),
    value = c(120, 145, 132, 158),
    sd = c(12, 15, 10, 14)
  )

  p <- ggplot(data, aes(x = category, y = value, fill = category)) +
    geom_bar(stat = "identity", alpha = 0.8, width = 0.7) +
    geom_errorbar(aes(ymin = value - sd, ymax = value + sd),
                  width = 0.2, size = 0.8) +
    geom_text(aes(label = value), vjust = -2, size = 5, fontface = "bold") +
    scale_fill_manual(values = c("#667eea", "#764ba2", "#f093fb", "#4facfe")) +
    labs(
      title = "Quarterly Performance with Error Bars",
      subtitle = "Values shown with standard deviation",
      x = "Quarter",
      y = "Performance Score"
    ) +
    theme(legend.position = "none") +
    ylim(0, 180)

  print(p)
  ggsave("bar_chart.png", width = 10, height = 6, dpi = 300)
  return(p)
}

# ========== 5. HEATMAP ==========
heatmap_plot <- function() {
  set.seed(123)
  data <- expand.grid(
    x = paste0("Var", 1:10),
    y = paste0("Cat", 1:8)
  )
  data$value <- rnorm(nrow(data), 50, 20)

  p <- ggplot(data, aes(x = x, y = y, fill = value)) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_gradient2(
      low = "#667eea",
      mid = "white",
      high = "#f093fb",
      midpoint = 50
    ) +
    labs(
      title = "Correlation Heatmap",
      subtitle = "Values represent correlation strength",
      x = "Variables",
      y = "Categories",
      fill = "Value"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  print(p)
  ggsave("heatmap.png", width = 10, height = 7, dpi = 300)
  return(p)
}

# ========== 6. VIOLIN PLOT ==========
violin_plot <- function() {
  set.seed(123)
  data <- data.frame(
    group = rep(c("Group A", "Group B", "Group C"), each = 100),
    value = c(rnorm(100, 50, 10), rnorm(100, 60, 12), rnorm(100, 55, 8))
  )

  p <- ggplot(data, aes(x = group, y = value, fill = group)) +
    geom_violin(alpha = 0.7, trim = FALSE) +
    geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
    scale_fill_manual(values = c("#667eea", "#764ba2", "#f093fb")) +
    labs(
      title = "Distribution Comparison with Violin Plots",
      subtitle = "Combined violin and box plots",
      x = "Group",
      y = "Value"
    ) +
    theme(legend.position = "none")

  print(p)
  ggsave("violin_plot.png", width = 10, height = 6, dpi = 300)
  return(p)
}

# ========== 7. FACETED PLOT ==========
faceted_plot <- function() {
  set.seed(123)
  data <- expand.grid(
    time = 1:50,
    category = c("Product A", "Product B", "Product C", "Product D")
  )
  data$value <- rnorm(nrow(data), 100, 20) +
    data$time * sample(c(0.5, 1, 1.5, 2), nrow(data), replace = TRUE)

  p <- ggplot(data, aes(x = time, y = value, color = category)) +
    geom_line(size = 1) +
    geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
    facet_wrap(~ category, ncol = 2, scales = "free_y") +
    scale_color_manual(values = c("#667eea", "#764ba2", "#f093fb", "#4facfe")) +
    labs(
      title = "Faceted Time Series Analysis",
      subtitle = "Individual trends by product category",
      x = "Time Period",
      y = "Value"
    ) +
    theme(legend.position = "none")

  print(p)
  ggsave("faceted_plot.png", width = 12, height = 8, dpi = 300)
  return(p)
}

# ========== MAIN FUNCTION ==========
generate_all_plots <- function() {
  cat("Generating R ggplot2 visualizations...\n\n")

  cat("1. Creating scatter plot...\n")
  scatter_plot()

  cat("2. Creating box plot...\n")
  box_plot()

  cat("3. Creating time series plot...\n")
  time_series_plot()

  cat("4. Creating bar chart...\n")
  bar_chart()

  cat("5. Creating heatmap...\n")
  heatmap_plot()

  cat("6. Creating violin plot...\n")
  violin_plot()

  cat("7. Creating faceted plot...\n")
  faceted_plot()

  cat("\n✓ All plots generated successfully!\n")
  cat("  PNG files saved in current directory.\n")
}

# Run all visualizations
generate_all_plots()
