# Statistical Modeling in R
# Advanced statistical analysis with multiple regression models

library(tidyverse)
library(caret)
library(MASS)
library(car)
library(lmtest)
library(glmnet)

# ===== Linear Regression Analysis =====
linear_regression_analysis <- function(data, formula) {
  # Fit model
  model <- lm(formula, data = data)

  # Summary statistics
  summary_stats <- summary(model)
  print(summary_stats)

  # Diagnostic plots
  par(mfrow = c(2, 2))
  plot(model)

  # Check assumptions
  # 1. Normality of residuals
  shapiro_test <- shapiro.test(residuals(model))
  print(paste("Shapiro-Wilk test p-value:", shapiro_test$p.value))

  # 2. Homoscedasticity (Breusch-Pagan test)
  bp_test <- bptest(model)
  print(paste("Breusch-Pagan test p-value:", bp_test$p.value))

  # 3. Multicollinearity (VIF)
  if (length(coef(model)) > 2) {
    vif_values <- vif(model)
    print("Variance Inflation Factors:")
    print(vif_values)
  }

  # 4. Autocorrelation (Durbin-Watson)
  dw_test <- dwtest(model)
  print(paste("Durbin-Watson test p-value:", dw_test$p.value))

  return(model)
}

# ===== Regularized Regression =====
ridge_lasso_regression <- function(X, y, alpha = 0.5) {
  # alpha = 0: Ridge
  # alpha = 1: Lasso
  # alpha in (0,1): Elastic Net

  # Cross-validation to find optimal lambda
  cv_model <- cv.glmnet(X, y, alpha = alpha, nfolds = 10)

  # Optimal lambda
  best_lambda <- cv_model$lambda.min

  # Fit model with best lambda
  model <- glmnet(X, y, alpha = alpha, lambda = best_lambda)

  # Coefficients
  coefs <- coef(model)

  # Plot coefficient paths
  plot(cv_model)

  list(
    model = model,
    cv_model = cv_model,
    best_lambda = best_lambda,
    coefficients = coefs
  )
}

# ===== Logistic Regression =====
logistic_regression_analysis <- function(data, formula) {
  # Fit logistic regression
  model <- glm(formula, data = data, family = binomial(link = "logit"))

  # Summary
  summary_stats <- summary(model)
  print(summary_stats)

  # Odds ratios
  odds_ratios <- exp(coef(model))
  print("Odds Ratios:")
  print(odds_ratios)

  # Confidence intervals
  conf_int <- confint(model)
  print("95% Confidence Intervals:")
  print(exp(conf_int))

  # Predictions
  predictions <- predict(model, type = "response")

  # ROC curve
  library(pROC)
  roc_obj <- roc(data$outcome, predictions)
  plot(roc_obj, main = "ROC Curve")
  auc_value <- auc(roc_obj)
  print(paste("AUC:", auc_value))

  return(model)
}

# ===== Polynomial Regression =====
polynomial_regression <- function(data, x_var, y_var, degree = 3) {
  # Create polynomial terms
  formula_str <- paste0(y_var, " ~ poly(", x_var, ", ", degree, ", raw = TRUE)")
  formula_obj <- as.formula(formula_str)

  # Fit model
  model <- lm(formula_obj, data = data)

  # Summary
  print(summary(model))

  # Plot
  plot_data <- data.frame(
    x = data[[x_var]],
    y = data[[y_var]]
  )

  # Predictions
  x_seq <- seq(min(data[[x_var]]), max(data[[x_var]]), length.out = 100)
  pred_data <- data.frame(x = x_seq)
  names(pred_data) <- x_var

  predictions <- predict(model, newdata = pred_data, interval = "confidence")

  # Visualization
  ggplot() +
    geom_point(data = plot_data, aes(x = x, y = y), alpha = 0.5) +
    geom_line(aes(x = x_seq, y = predictions[, "fit"]), color = "blue", size = 1) +
    geom_ribbon(aes(x = x_seq,
                    ymin = predictions[, "lwr"],
                    ymax = predictions[, "upr"]),
                alpha = 0.2, fill = "blue") +
    labs(title = paste("Polynomial Regression (degree =", degree, ")"),
         x = x_var, y = y_var) +
    theme_minimal()
}

# ===== Mixed Effects Models =====
mixed_effects_model <- function(data, formula, random_formula) {
  library(lme4)

  # Fit mixed model
  model <- lmer(formula, data = data, REML = TRUE)

  # Summary
  print(summary(model))

  # Random effects
  random_effects <- ranef(model)
  print("Random Effects:")
  print(random_effects)

  # Fixed effects
  fixed_effects <- fixef(model)
  print("Fixed Effects:")
  print(fixed_effects)

  # Residual plots
  plot(model)

  # R-squared
  library(MuMIn)
  r_squared <- r.squaredGLMM(model)
  print("R-squared:")
  print(r_squared)

  return(model)
}

# ===== Generalized Additive Models =====
gam_analysis <- function(data, formula) {
  library(mgcv)

  # Fit GAM
  model <- gam(formula, data = data)

  # Summary
  print(summary(model))

  # Visualization
  plot(model, pages = 1, residuals = TRUE)

  # Diagnostics
  gam.check(model)

  return(model)
}

# ===== Model Comparison =====
model_comparison <- function(...) {
  models <- list(...)

  # AIC comparison
  aic_values <- sapply(models, AIC)
  names(aic_values) <- paste0("Model", seq_along(models))

  # BIC comparison
  bic_values <- sapply(models, BIC)
  names(bic_values) <- paste0("Model", seq_along(models))

  # ANOVA
  anova_result <- anova(..., test = "Chisq")

  list(
    AIC = aic_values,
    BIC = bic_values,
    ANOVA = anova_result
  )
}

# ===== Example Usage =====
example_analysis <- function() {
  # Load built-in dataset
  data(mtcars)

  # 1. Linear Regression
  cat("\n=== Linear Regression ===\n")
  lm_model <- linear_regression_analysis(
    mtcars,
    mpg ~ wt + hp + qsec
  )

  # 2. Ridge/Lasso Regression
  cat("\n=== Ridge/Lasso Regression ===\n")
  X <- as.matrix(mtcars[, c("wt", "hp", "qsec")])
  y <- mtcars$mpg
  reg_model <- ridge_lasso_regression(X, y, alpha = 0.5)

  # 3. Polynomial Regression
  cat("\n=== Polynomial Regression ===\n")
  poly_plot <- polynomial_regression(mtcars, "wt", "mpg", degree = 3)
  print(poly_plot)

  # 4. GAM
  cat("\n=== Generalized Additive Model ===\n")
  gam_model <- gam_analysis(
    mtcars,
    mpg ~ s(wt) + s(hp) + qsec
  )

  # 5. Model Comparison
  cat("\n=== Model Comparison ===\n")
  comparison <- model_comparison(lm_model, gam_model)
  print(comparison)
}

# Run example
if (sys.nframe() == 0) {
  example_analysis()
}
