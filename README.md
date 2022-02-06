From MiLB to MLB Debut: a Machine Learning Application
================
Chung-Hao Lee
02/10/2022

<!-- README.md is generated from "From-MiLB-to-MLB-Debut-a-Machine-Learning-Application.Rmd". Please edit that file --> 

``` r
#loading libraries
library("tidyverse") # for data processing
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✓ ggplot2 3.3.5     ✓ purrr   0.3.4
    ## ✓ tibble  3.1.6     ✓ dplyr   1.0.7
    ## ✓ tidyr   1.1.4     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library("tidymodels") # for the resample, recipe, workflow package
```

    ## Registered S3 method overwritten by 'tune':
    ##   method                   from   
    ##   required_pkgs.model_spec parsnip

    ## ── Attaching packages ────────────────────────────────────── tidymodels 0.1.4 ──

    ## ✓ broom        0.7.11     ✓ rsample      0.1.1 
    ## ✓ dials        0.0.10     ✓ tune         0.1.6 
    ## ✓ infer        1.0.0      ✓ workflows    0.2.4 
    ## ✓ modeldata    0.1.1      ✓ workflowsets 0.1.0 
    ## ✓ parsnip      0.1.7      ✓ yardstick    0.0.8 
    ## ✓ recipes      0.1.17

    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## x scales::discard() masks purrr::discard()
    ## x dplyr::filter()   masks stats::filter()
    ## x recipes::fixed()  masks stringr::fixed()
    ## x dplyr::lag()      masks stats::lag()
    ## x yardstick::spec() masks readr::spec()
    ## x recipes::step()   masks stats::step()
    ## • Use suppressPackageStartupMessages() to eliminate package startup messages

``` r
library("parsnip") # for the XGB, RF and DT models
library("kernlab") # for the SVM model
```

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:scales':
    ## 
    ##     alpha

    ## The following object is masked from 'package:purrr':
    ## 
    ##     cross

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

``` r
library("glmnet") # for lasso regression
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 4.1-1

``` r
knitr::opts_chunk$set(
  collapse = TRUE, 
  comment = "#>",
  fig.path = "fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-",
  out.width = "100%",
  message=FALSE, 
  warning=FALSE
)
```

``` r
#loading dataset and do data preparation
df_mlb <-
  read_csv("/Users/yginger/Desktop/Maryland/GA/Adam/Fall 2021/MLB_debut_prediction/data/mlb_draft_01to10.csv") %>% 
  mutate(mlb_debut = as.factor(mlb_debut),
         sch_reg = as.factor(sch_reg),
         birth_place = as.factor(birth_place),
         team = as.factor(team),
         position = as.factor(position),
         schooltype = as.factor(schooltype),
         bats = as.factor(bats),
         throws = as.factor(throws),
         bmi = weight/(height/100)^2,
         hr_ab = hr/ab,
         iso = slg - avg,
         bb_so = bb/so,
         sbr = sb/(sb+cs)) %>% 
  rename(age = age_at_draft,
         overall_pick = draft_overall,
         round = draft_round,
         year = draft_year,
         b2 = dbl,
         b3 = tpl
         ) 
  
```

``` r
#Turn every na to 0 in sbr column
df_mlb$sbr <- replace_na(df_mlb$sbr, 0)
```

``` r
# Calculate proportion of each category
df_mlb %>% 
  count(mlb_debut) %>% 
  mutate(prop = n/sum(n))
#> # A tibble: 2 × 3
#>   mlb_debut     n  prop
#>   <fct>     <int> <dbl>
#> 1 no         3588 0.830
#> 2 yes         733 0.170
```

# EDA

``` r
### Distributed by teams with percentage
df_mlb_team <-
  df_mlb %>%
  group_by(team, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(team) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(team_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -team_prop)


ggplot(df_mlb_team)+
  geom_col(mapping = aes(x = team, y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_point(mapping = aes(x = factor(team), y = team_prop*150), color = 'black', alpha = 0.5, size = 1) +
  geom_text(mapping = aes(x = factor(team), y = team_prop*150, label = team_prop), vjust = -1, size = 2.5)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 150, name = "MLB debut percentage")) +
  labs(x = "Team", y = "Number of players", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-6-1.png" width="100%" />

``` r
### Distributed by positions
df_mlb_position <-
  df_mlb %>%
  group_by(position, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(position) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(positions_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -positions_prop)


ggplot(df_mlb_position)+
  geom_col(mapping = aes(x = factor(position, level = c('IF', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF')), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_point(mapping = aes(x = factor(position, level = c('IF', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF')), y = positions_prop*1250), color = 'black', alpha = 0.5, size = 1) +
  geom_text(mapping = aes(x = factor(position, level = c('IF', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF')), y = positions_prop*1250, label = positions_prop), vjust = -1, alpha = 0.8)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ ./1250 , name = "MLB debut percentage"), limits = c(0,1200)) +
  labs(x = "Position", y = "Number of players", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-7-1.png" width="100%" />

``` r
### Distributed by ages with percentage
df_mlb_age <-
  df_mlb %>%
  group_by(age, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(age) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(age_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -age_prop)


ggplot(df_mlb_age)+
  geom_col(mapping = aes(x = as.factor(age), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(age), y = age_prop*1250, group = 1), color = 'black') +
  geom_point(mapping = aes(x = as.factor(age), y = age_prop*1250), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(age), y = age_prop*1250, label = age_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ ./1250 , name = "MLB debut percentage"), limits = c(0,1200)) +
  labs(x = "Age", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-8-1.png" width="100%" />

``` r
### Distributed by bats with percentage
df_mlb_bats <-
  df_mlb %>%
  group_by(bats, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(bats) %>%
  mutate(bats_prop = round(n / sum(n), digits = 2)) %>%
  filter(mlb_debut == "yes") 


ggplot()+
  geom_bar(data = df_mlb, mapping = aes(x = bats, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_point(data = df_mlb_bats, mapping = aes(x = bats, y = bats_prop*2500), color = 'black') +
  geom_text(data = df_mlb_bats, mapping = aes(x = bats, y = bats_prop*2500, label = bats_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 2500, name = "MLB debut percentage"), limits = c(0,2400)) +
  labs(x = "Bats", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-9-1.png" width="100%" />

``` r
### Distributed by throws with percentage
df_mlb_throw <-
  df_mlb %>%
  group_by(throws, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(throws) %>%
  mutate(throw_prop = round(n / sum(n), digits = 2)) %>%
  filter(mlb_debut == "yes") 


ggplot()+
  geom_bar(data = df_mlb, mapping = aes(x = throws, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_point(data = df_mlb_throw, mapping = aes(x = throws, y = throw_prop*3500), color = 'black') +
  geom_text(data = df_mlb_throw, mapping = aes(x = throws, y = throw_prop*3500, label = throw_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 3500, name = "MLB debut percentage"), limits = c(0,3500), n.breaks = 7) +
  labs(x = "Throws", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-10-1.png" width="100%" />

``` r
### Distributed by year with percentage
df_mlb_year <-
  df_mlb %>%
  group_by(year, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(year) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(year_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -year_prop)


ggplot(df_mlb_year)+
  geom_col( mapping = aes(x = as.factor(year), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(year), y = year_prop*400, group = 1), color = 'black') +
  geom_point(mapping = aes(x = as.factor(year), y = year_prop*400), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(year), y = year_prop*400, label = year_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 400, name = "MLB debut percentage")) +
  labs(x = "Draft Year", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-11-1.png" width="100%" />

``` r
### Distributed by height with percentage
df_mlb_height <-
  df_mlb %>%
  mutate(height_group = ifelse(height<170, "170-", ifelse(height>=170 & height <175, "170-174", ifelse(height>=175 & height <180, "175-180", ifelse(height>=180 & height <185, "180-185", ifelse(height>=185 & height <190, "185-190", ifelse(height>=190 & height <195, "190-195", ifelse(height>=195 & height < 200, "195-200", "200+")))))))) %>% 
  group_by(height_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(height_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(height_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -height_prop)


ggplot(df_mlb_height)+
  geom_col( mapping = aes(x = as.factor(height_group), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(height_group), y = height_prop*1500, group = 1), color = 'black') +
  geom_point(mapping = aes(x = as.factor(height_group), y = height_prop*1500), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(height_group), y = height_prop*1500, label = height_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1500, name = "MLB debut percentage"), limits = c(0,1500), breaks = c(0, 375, 750, 1125, 1500)) +
  labs(x = "Height", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-12-1.png" width="100%" />

``` r
### Distributed by weight with percentage (new ver.)
df_mlb_weight <-
  df_mlb %>%
  mutate(weight_group = ifelse(weight>=60 & weight <70, "60-69", ifelse(weight>=70 & weight <80, "70-79", ifelse(weight>=80 & weight <90, "80-89", ifelse(weight>=90 & weight <100, "90-99", ifelse(weight>=100 & weight <110, "100-109", ifelse(weight>=110 & weight < 120, "110-119", "120+"))))))) %>% 
  group_by(weight_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(weight_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(weight_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -weight_prop)



ggplot(df_mlb_weight)+
  geom_col( mapping = aes(x = factor(weight_group, level = c("60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120+")), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = factor(weight_group, level = c("60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120+")), y = weight_prop*1600, group = 1), color = 'black') +
  geom_point(mapping = aes(x = factor(weight_group, level = c("60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120+")), y = weight_prop*1600), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = factor(weight_group, level = c("60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120+")), y = weight_prop*1600, label = weight_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1600, name = "MLB debut percentage"), limits = c(0,1600), breaks = c(0, 400, 800, 1200, 1600)) +
  labs(x = "Weight", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-13-1.png" width="100%" />

``` r
### Distributed by bmi with percentage
df_mlb_bmi <-
  df_mlb %>%
  mutate(bmi_group = ifelse(bmi>=20 & bmi <22, "20-22", ifelse(bmi>=22 & bmi <24, "22-24", ifelse(bmi>=24 & bmi <26, "24-26", ifelse(bmi>=26 & bmi <28, "26-28", ifelse(bmi>=28 & bmi <30, "28-30", "30+")))))) %>% 
  group_by(bmi_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(bmi_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(bmi_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -bmi_prop)


ggplot(df_mlb_bmi)+
  geom_col( mapping = aes(x = as.factor(bmi_group), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(bmi_group), y = bmi_prop*1500, group = 1), color = 'black') +
  geom_point(mapping = aes(x = as.factor(bmi_group), y = bmi_prop*1500), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(bmi_group), y = bmi_prop*1500, label = bmi_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1500, name = "MLB debut percentage"), limits = c(0,1500), n.breaks = 5) +
  labs(x = "BMI", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-14-1.png" width="100%" />

``` r
### Distributed by avg with percentage
df_mlb_avg <-
  df_mlb %>%
  mutate(avg_group = ifelse(avg <0.1, "100-", ifelse(avg>=0.1 & avg <0.15, "100-150", ifelse(avg>=0.15 & avg <0.2, "150-200", ifelse(avg>=0.2 & avg <0.25, "200-250", ifelse(avg>=0.25 & avg <0.3, "250-300", "300+")))))) %>% 
  group_by(avg_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(avg_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(avg_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -avg_prop)



ggplot(df_mlb_avg)+
  geom_col( mapping = aes(x = factor(avg_group, level = c("100-", "100-150", "150-200", "200-250", "250-300", "300+")), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = factor(avg_group, level = c("100-", "100-150", "150-200", "200-250", "250-300", "300+")), y = avg_prop*1600, group = 1), color = 'black') +
  geom_point(mapping = aes(x = factor(avg_group, level = c("100-", "100-150", "150-200", "200-250", "250-300", "300+")), y = avg_prop*1600), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = factor(avg_group, level = c("100-", "100-150", "150-200", "200-250", "250-300", "300+")), y = avg_prop*1600, label = avg_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1600, name = "MLB debut percentage"), limits = c(0,1600), breaks = c(0, 400, 800, 1200, 1600)) +
  labs(x = "AVG", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-15-1.png" width="100%" />

``` r
### Distributed by obp with percentage
df_mlb_obp <-
  df_mlb %>%
  mutate(obp_group = ifelse(obp <0.2, "200-", ifelse(obp>=0.2 & obp <0.25, "200-250", ifelse(obp>=0.25 & obp <0.3, "250-300", ifelse(obp>=0.3 & obp <0.35, "300-350", ifelse(obp>=0.35 & obp <0.4, "350-400", "400+")))))) %>% 
  group_by(obp_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(obp_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(obp_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -obp_prop)


ggplot(df_mlb_obp)+
  geom_col( mapping = aes(x = as.factor(obp_group), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(obp_group), y = obp_prop*2000, group = 1), color = 'black') +
  geom_point( mapping = aes(x = as.factor(obp_group), y = obp_prop*2000), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(obp_group), y = obp_prop*2000, label = obp_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 2000, name = "MLB debut percentage")) +
  labs(x = "OBP", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-16-1.png" width="100%" />

``` r
### Distributed by slg with percentage
df_mlb_slg <-
  df_mlb %>%
  mutate(slg_group = ifelse(slg <0.3, "300-", ifelse(slg>=0.3 & slg <0.35, "300-350", ifelse(slg>=0.35 & slg <0.4, "350-400", ifelse(slg>=0.4 & slg <0.45, "400-450", ifelse(slg>=0.45 & slg <0.5, "450-500", "500+")))))) %>%  
  group_by(slg_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(slg_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(slg_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -slg_prop)


ggplot(df_mlb_slg)+
  geom_col( mapping = aes(x = as.factor(slg_group), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = as.factor(slg_group), y = slg_prop*1200, group = 1), color = 'black') +
  geom_point(mapping = aes(x = as.factor(slg_group), y = slg_prop*1200), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = as.factor(slg_group), y = slg_prop*1200, label = slg_prop), vjust = -1)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1200, name = "MLB debut percentage"), limits = c(0,1200), breaks = c(0, 300, 600, 900, 1200)) +
  labs(x = "SLG", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-17-1.png" width="100%" />

``` r
### Distributed by iso with percentage new
df_mlb_iso <-
  df_mlb %>%
  mutate(iso_group = ifelse(iso <0.1, "0-100", ifelse(iso>=0.1 & iso <0.125, "100-125", ifelse(iso>=0.125 & iso <0.15, "125-150", ifelse(iso>=0.15 & iso <0.175, "150-175", ifelse(iso>=0.175 & iso <0.2, "175-200", "200+")))))) %>%  
  group_by(iso_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(iso_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(iso_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -iso_prop)


ggplot(df_mlb_iso)+
  geom_col( mapping = aes(x = factor(iso_group, level = c("0-100", "100-125", "125-150", "150-175", "175-200", "200+")), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(mapping = aes(x = factor(iso_group, level = c("0-100", "100-125", "125-150", "150-175", "175-200", "200+")), y = iso_prop*1800, group = 1), color = 'black') +
  geom_point(mapping = aes(x = factor(iso_group, level = c("0-100", "100-125", "125-150", "150-175", "175-200", "200+")), y = iso_prop*1800), color = 'black', size = 0.6) +
  geom_text(mapping = aes(x = factor(iso_group, level = c("0-100", "100-125", "125-150", "150-175", "175-200", "200+")), y = iso_prop*1800, label = iso_prop), vjust = -0.5)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1800, name = "MLB debut percentage"), limits = c(0,1750), breaks = c(0, 450, 900, 1350, 1800)) +
  labs(x = "ISO", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-18-1.png" width="100%" />

``` r
### Distributed by ops with percentage
df_mlb_ops <-
  df_mlb %>%
  mutate(ops_group = ifelse(ops <0.5, "500-", ifelse(ops>=0.5 & ops <0.6, "500-600", ifelse(ops>=0.6 & ops <0.7, "600-700", ifelse(ops>=0.7 & ops <0.8, "700-800", ifelse(ops>=0.8 & ops <0.9, "800-900", "900+")))))) %>%  
  group_by(ops_group, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(ops_group) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(ops_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -ops_prop)

ggplot(df_mlb_ops)+
  geom_col( mapping = aes(x = as.factor(ops_group), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(data = df_mlb_ops, mapping = aes(x = as.factor(ops_group), y = ops_prop*1500, group = 1), color = 'black') +
  geom_point(data = df_mlb_ops, mapping = aes(x = as.factor(ops_group), y = ops_prop*1500), color = 'black', size = 0.6) +
  geom_text(data = df_mlb_ops, mapping = aes(x = as.factor(ops_group), y = ops_prop*1500, label = ops_prop), vjust = -0.5)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 1500, name = "MLB debut percentage"), limits = c(0,1500), n.breaks = 5) +
  labs(x = "OPS", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-19-1.png" width="100%" />

``` r
### Among players who make MLB debut, what's their ops distribution by position  
df_mlb %>% 
  filter(mlb_debut == 'yes',
         position == c("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF")) %>% 
  ggplot()+
  geom_density(mapping = aes(x = ops, color = position), binwidth = 0.05)+
  labs(x = "OPS", y ="Frequency", fill = "Position")+
  theme_classic()+
  theme(axis.text.x=element_text(angle=60, hjust=1))+
  scale_color_brewer(palette = "Paired")
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-20-1.png" width="100%" />

``` r

df_mlb %>% 
  mutate(ops_group = ifelse(ops <0.5, "500-", ifelse(ops>=0.5 & ops <0.6, "500-600", ifelse(ops>=0.6 & ops <0.7, "600-700", ifelse(ops>=0.7 & ops <0.8, "700-800", ifelse(ops>=0.8 & ops <0.9, "800-900", "900+")))))) %>% 
  group_by(ops_group, position, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(ops_group, position) %>%
  mutate(ops_pos_prop = n / sum(n)) %>% 
  filter(mlb_debut == 'yes')
#> # A tibble: 35 × 5
#> # Groups:   ops_group, position [35]
#>    ops_group position mlb_debut     n ops_pos_prop
#>    <chr>     <fct>    <fct>     <int>        <dbl>
#>  1 500-600   C        yes           1      0.00730
#>  2 600-700   2B       yes           6      0.0353 
#>  3 600-700   3B       yes           5      0.0352 
#>  4 600-700   C        yes          36      0.129  
#>  5 600-700   CF       yes           2      0.0667 
#>  6 600-700   IF       yes           2      0.0312 
#>  7 600-700   LF       yes           2      0.0392 
#>  8 600-700   OF       yes          13      0.0309 
#>  9 600-700   SS       yes          14      0.0765 
#> 10 700-800   1B       yes          22      0.103  
#> # … with 25 more rows
```

``` r
### Among players who make MLB debut, what's their ops distribution by position  
df_mlb %>% 
  mutate(ops_group = ifelse(ops <0.5, "500-", ifelse(ops>=0.5 & ops <0.6, "500-600", ifelse(ops>=0.6 & ops <0.7, "600-700", ifelse(ops>=0.7 & ops <0.8, "700-800", ifelse(ops>=0.8 & ops <0.9, "800-900", "900+")))))) %>% 
  filter(mlb_debut == 'yes',
         position == c("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF")) %>% 
  ggplot()+
  geom_bar(mapping = aes(x = ops_group, fill = position), position = position_dodge2(width = 1, preserve = "single"))+
  labs(x = "OPS", y ="Number of players", fill = "Position")+
  theme_classic()+
  theme(axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_brewer(palette = "Paired")
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-21-1.png" width="100%" />

``` r

df_mlb %>% 
  mutate(ops_group = ifelse(ops <0.5, "500-", ifelse(ops>=0.5 & ops <0.6, "500-600", ifelse(ops>=0.6 & ops <0.7, "600-700", ifelse(ops>=0.7 & ops <0.8, "700-800", ifelse(ops>=0.8 & ops <0.9, "800-900", "900+")))))) %>% 
  group_by(ops_group, position, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(ops_group, position) %>%
  mutate(ops_pos_prop = n / sum(n)) %>% 
  filter(mlb_debut == 'yes')
#> # A tibble: 35 × 5
#> # Groups:   ops_group, position [35]
#>    ops_group position mlb_debut     n ops_pos_prop
#>    <chr>     <fct>    <fct>     <int>        <dbl>
#>  1 500-600   C        yes           1      0.00730
#>  2 600-700   2B       yes           6      0.0353 
#>  3 600-700   3B       yes           5      0.0352 
#>  4 600-700   C        yes          36      0.129  
#>  5 600-700   CF       yes           2      0.0667 
#>  6 600-700   IF       yes           2      0.0312 
#>  7 600-700   LF       yes           2      0.0392 
#>  8 600-700   OF       yes          13      0.0309 
#>  9 600-700   SS       yes          14      0.0765 
#> 10 700-800   1B       yes          22      0.103  
#> # … with 25 more rows
```

``` r
### Distributed by rounds with percentage
df_mlb_round <-
  df_mlb %>%
  group_by(round, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(round) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(round_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -round_prop) 


ggplot(df_mlb_round)+
  geom_col(mapping = aes(x = as.factor(round), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_line(data = . %>% filter(round <=7), mapping = aes(x = round, y = round_prop*160, group = 1), color = '#663300', width = 3, alpha = 0.8) +
  geom_point(data = . %>% filter(round <= 7), mapping = aes(x = as.factor(round), y = round_prop*160), color = 'black', alpha = 0.8, size = 0.1) +
  geom_text(data = . %>% filter(round <= 7), mapping = aes(x = as.factor(round), y = round_prop*160, label = round_prop), vjust = -0.6, hjust = -0.05, alpha = 0.5)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 160, name = "MLB debut percentage"), limits = c(0,160), breaks = c(0, 40, 80, 120, 160)) +
  labs(x = "Round", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-22-1.png" width="100%" />

``` r
### Distributed by education type
df_mlb_edutype <-
  df_mlb %>%
  group_by(schooltype, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(schooltype) %>%
  spread(mlb_debut, n, fill = 0) %>%
  mutate(edutype_prop = round(yes / (yes+no), 2)) %>%
  gather(key = "mlb_debut", value = "count", no, yes, -edutype_prop) 

ggplot(df_mlb_edutype)+
  geom_col(mapping = aes(x = as.factor(schooltype), y = count, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  geom_point(mapping = aes(as.factor(schooltype), y = edutype_prop*2500), color = 'black') +
  geom_text(mapping = aes(as.factor(schooltype), y = edutype_prop*2500, label = edutype_prop), vjust = -0.6, hjust = -0.05)+
  scale_y_continuous("Number of players", sec.axis = sec_axis(trans =  ~ . / 2500, name = "MLB debut percentage")) +
  labs(x = "Last School Type", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=60, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-23-1.png" width="100%" />

``` r
### Distributed by school state 
df_mlb %>% 
  ggplot()+
  geom_bar(mapping = aes(x = as.factor(sch_reg), fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  labs(x = "Last School Place", y = "Number of players", fill = "MLB")+
  ggthemes::theme_hc()+
  scale_y_continuous(breaks=c(0, 100, 200, 300, 400, 500, 600, 700), limits=c(0, 700))+
  theme(legend.position="right", axis.text.x=element_text(angle=70, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-24-1.png" width="100%" />

``` r

df_mlb %>% 
  group_by(sch_reg, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(sch_reg) %>%
  mutate(state_prop = n / sum(n)) %>% 
  filter(mlb_debut == 'yes') %>% 
  arrange(desc(state_prop))
#> # A tibble: 44 × 4
#> # Groups:   sch_reg [44]
#>    sch_reg mlb_debut     n state_prop
#>    <fct>   <fct>     <int>      <dbl>
#>  1 NE      yes           8      0.267
#>  2 AZ      yes          34      0.266
#>  3 NV      yes          12      0.24 
#>  4 LA      yes          30      0.227
#>  5 MS      yes          18      0.207
#>  6 BC      yes           3      0.2  
#>  7 GA      yes          34      0.2  
#>  8 ON      yes           3      0.2  
#>  9 SC      yes          24      0.2  
#> 10 OK      yes          25      0.197
#> # … with 34 more rows
```

``` r
### Distributed by birth place 
df_mlb %>% 
  ggplot()+
  geom_bar(mapping = aes(x = as.factor(birth_place), fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  labs(x = "Birth Place", y = "Number of players", fill = "MLB")+
  ggthemes::theme_hc()+
  scale_y_continuous(breaks=c(0, 200, 400, 600, 800), limits=c(0, 800))+
  theme(legend.position="right", axis.text.x=element_text(angle=90, size = 7))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-25-1.png" width="100%" />

``` r

df_mlb %>% 
  group_by(birth_place, mlb_debut) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(birth_place) %>%
  mutate(bir_prop = n / sum(n)) %>% 
  filter(mlb_debut == 'yes') %>% 
  arrange(desc(bir_prop))
#> # A tibble: 62 × 4
#> # Groups:   birth_place [62]
#>    birth_place mlb_debut     n bir_prop
#>    <fct>       <fct>     <int>    <dbl>
#>  1 NL          yes           1    1    
#>  2 VT          yes           2    0.667
#>  3 BR          yes           1    0.5  
#>  4 GE          yes           1    0.5  
#>  5 SR          yes           1    0.5  
#>  6 VG          yes           2    0.4  
#>  7 WY          yes           1    0.333
#>  8 NH          yes           2    0.286
#>  9 CT          yes           7    0.269
#> 10 OR          yes          10    0.263
#> # … with 52 more rows
```

``` r
### Distributed by School
df_mlb %>% 
  ggplot()+
  geom_bar(mapping = aes(x = as.factor(school), fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"))+
  labs(x = "Last School Attended", y = "Number of players", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right", axis.text.x=element_text(angle=90, size = 3))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-26-1.png" width="100%" />

``` r
### AVG over year
df_mlb %>% 
  ggplot()+
  geom_boxplot(mapping = aes(x = as.factor(year), y = avg, fill = mlb_debut), coef = 5)+
  labs(x = "Draft Year", y = "AVG", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-27-1.png" width="100%" />

``` r
### OBP over year
df_mlb %>% 
  ggplot()+
  geom_boxplot(mapping = aes(x = as.factor(year), y = obp, fill = mlb_debut), coef =5)+
  labs(x = "Draft Year", y = "OBP", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-28-1.png" width="100%" />

``` r
### SLG over year
df_mlb %>% 
  ggplot()+
  geom_boxplot(mapping = aes(x = as.factor(year), y = slg, fill = mlb_debut), coef =5)+
  labs(x = "Draft Year", y = "SLG", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-29-1.png" width="100%" />

``` r
### OPS over year
df_mlb %>% 
  ggplot()+
  geom_boxplot(mapping = aes(x = as.factor(year), y = ops, fill = mlb_debut), coef =5)+
  labs(x = "Draft Year", y = "OPS", fill = "MLB")+
  ggthemes::theme_hc()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-30-1.png" width="100%" />

``` r
### Age vs round
df_mlb %>% 
  ggplot()+
  geom_smooth(mapping = aes(x = round, y = as.numeric(age), fill = mlb_debut), method = 'loess', color = "black")+
  labs(x = "Round", y = "Age", fill = "MLB")+
  scale_x_continuous(breaks = c(1:50))+
  theme_classic()+
  theme(legend.position="right", axis.text.x=element_text(angle=70, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-31-1.png" width="100%" />

``` r
#TTO vs ISO
df_mlb %>% 
  mutate(iso = slg - avg,
         iso_group = ifelse(iso <0.05, "0-50", ifelse(iso>=0.05 & iso <0.1, "50-100", ifelse(iso>=0.1 & iso <0.15, "100-150", ifelse(iso>=0.15 & iso <0.2, "150-200", ifelse(iso>=0.2 & iso <0.25, "200-250", "250+")))))) %>% 
  ggplot(aes(x = factor(iso_group, level = c("0-50", "50-100", "100-150", "150-200", "200-250", "250+"))))+
  geom_boxplot(mapping = aes( y = (hr+bb+so)/ab, fill = mlb_debut), position = position_dodge2(width = 1, preserve = "single"), coef = 5)+
  labs(x = "ISO", y = "(HR+BB+SO)/AB", fill = "MLB")+
  theme_light()+
  theme(axis.text.x=element_text(angle=70, hjust=1))+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-32-1.png" width="100%" />

``` r
#TTO through year 01 - 10
df_mlb %>% 
  ggplot()+
  geom_smooth(mapping = aes(x = year, y = (hr+bb+so)/g, fill = mlb_debut), color = 'black',  method = "loess")+
  labs(x = "Year", y = "(HR+BB+SO)/G", fill = "MLB")+
  scale_x_continuous(breaks = c(2001:2010))+
  theme_classic()+
  theme(legend.position="right")+
  scale_fill_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-33-1.png" width="100%" />

# Variable selection

#### Lasso for all variables

``` r
#Prepare dataset for lasso for all variables
df_mlb_lasso <- df_mlb %>% 
  select(-name, -highLevel, -school) %>% 
  mutate(team = as.numeric(team),
         bats = as.numeric(bats),
         throws = as.numeric(throws),
         mlb_debut = as.numeric(mlb_debut),
         schooltype = as.numeric(schooltype),
         birth_place = as.numeric(birth_place),
         sch_reg = as.numeric(sch_reg),
         position = as.numeric(position))

#Do normalization(0-1)
df_mlb_lasso[c(1:12, 14:44)] = BBmisc::normalize(df_mlb_lasso[c(1:12, 14:44)], method = "range", range = c(0, 1))
```

``` r
df_mlb_lasso <- df_mlb_lasso %>% 
  mutate(mlb_debut = as.factor(mlb_debut))

x = model.matrix(mlb_debut~., df_mlb_lasso)[,-1]
                                        
y = df_mlb_lasso %>%
  select(mlb_debut) %>%
  unlist() %>%
  as.numeric()
```

``` r
set.seed(123)
lasso = glmnet(x = x, 
               y = y, 
               alpha = 1,
               family = "binomial")

plot(lasso, xvar='lambda', main="Lasso")
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-36-1.png" width="100%" />

``` r
cv.lasso = cv.glmnet(x = x, 
                     y = y, 
                     alpha = 1,  # lasso
                     family = "binomial")
```

``` r
#Selecting the best lambda
best.lambda_lasso4all = cv.lasso$lambda.min
best.lambda_lasso4all
#> [1] 0.0003978768
```

``` r
#Plot the Lasso regression
plot(lasso, xvar='lambda', main="Lasso")
abline(v=log(best.lambda_lasso4all), col="blue", lty=5.5 )
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-39-1.png" width="100%" />

``` r
#List out all variables and their coefficients
coef(cv.lasso, s = "lambda.min")
#> 44 x 1 sparse Matrix of class "dgCMatrix"
#>                          1
#> (Intercept)  -1.730596e+01
#> position      2.992724e-02
#> height        1.551053e+00
#> weight        .           
#> bats          1.500362e-01
#> throws        1.212961e-01
#> birth_year    2.842738e-01
#> birth_place  -2.282719e-01
#> age          -1.415539e+00
#> year          .           
#> round         .           
#> overall_pick -3.603874e+00
#> team          2.517641e-01
#> schooltype   -6.691464e-02
#> sch_reg       8.168702e-02
#> g             .           
#> ab            9.240016e+00
#> r             .           
#> h             8.071677e-05
#> b2            .           
#> b3            6.822985e-01
#> hr            1.619119e+00
#> rbi          -1.422723e+01
#> sb            8.299497e+00
#> cs           -9.278972e+00
#> bb            .           
#> so            5.435552e+00
#> sh            4.613714e+00
#> sf            3.544626e+00
#> hbp          -1.750168e+00
#> gdp           3.394620e+00
#> tb            6.376811e-02
#> pa            .           
#> xbh           2.516271e+00
#> avg           1.595021e+01
#> obp           .           
#> slg           .           
#> ops           1.176238e+01
#> babip        -8.959748e+00
#> bmi           2.383566e+00
#> hr_ab         .           
#> iso           .           
#> bb_so         5.084953e-01
#> sbr          -9.353736e-01
```

#### Lasso for pre-selected variables

``` r
#Prepare dataset for lasso for pre-selected variables
df_mlb_lasso_s <- df_mlb %>% 
  select(bats, age, bmi, round, overall_pick, team, avg, obp, slg, ops, iso, bb_so, hr_ab, sbr, ab, mlb_debut) %>% 
  mutate(team = as.numeric(team),
         bats = as.numeric(bats),
         mlb_debut = as.numeric(mlb_debut))

#Do normalization(0-1)
df_mlb_lasso_s[c(1:15)] = BBmisc::normalize(df_mlb_lasso_s[c(1:15)], method = "range", range = c(0, 1))
```

``` r
df_mlb_lasso_s <- df_mlb_lasso_s %>% 
  mutate(mlb_debut = as.factor(mlb_debut))

x = model.matrix(mlb_debut~., df_mlb_lasso_s)[,-1]

y = df_mlb_lasso_s %>%
  select(mlb_debut) %>%
  unlist() %>%
  as.numeric()
```

``` r
set.seed(123)
lasso = glmnet(x = x, 
               y = y, 
               alpha = 1,
               family = "binomial")

plot(lasso, xvar='lambda', main="Lasso")
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-43-1.png" width="100%" />

``` r
cv.lasso = cv.glmnet(x = x, 
                     y = y, 
                     alpha = 1,  # lasso
                     family = "binomial")
```

``` r
#Selecting the best lambda
best.lambda_lasso4lch = cv.lasso$lambda.min
best.lambda_lasso4lch
#> [1] 0.0003808978
```

``` r
#Plot the Lasso regression
plot(lasso, xvar='lambda', main="Lasso")
abline(v=log(best.lambda_lasso4lch), col="blue", lty=5.5 )
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-46-1.png" width="100%" />

``` r
# List out all variables and their coefficients
coef(cv.lasso, s = "lambda.min")
#> 16 x 1 sparse Matrix of class "dgCMatrix"
#>                        1
#> (Intercept)  -14.7670400
#> bats           0.3192117
#> age           -1.0935613
#> bmi            1.3408020
#> round          .        
#> overall_pick  -3.8509179
#> team           0.3093437
#> avg            9.1094086
#> obp            .        
#> slg            3.4300193
#> ops            2.8367939
#> iso            .        
#> bb_so          2.8306870
#> hr_ab          .        
#> sbr            0.7763358
#> ab            10.4967644
```

# Modeling

## XGBoost

``` r
# Prepare dataset for XGBoost
df_mlb_xg <- df_mlb %>% 
  select(-name, -highLevel, -school) %>% # Remove variables of name, highlevel and school
  fastDummies::dummy_cols(select_columns = c( 'team', 'schooltype', 'bats', 'throws', 'position', 'sch_reg', 'birth_place'), remove_selected_columns = TRUE) # Converting the categorical variable into dummies

str(df_mlb_xg)
#> tibble [4,321 × 222] (S3: tbl_df/tbl/data.frame)
#>  $ height              : num [1:4321] 191 191 183 175 183 188 185 183 191 191 ...
#>  $ weight              : num [1:4321] 95 99 82 86 86 104 86 98 86 88 ...
#>  $ birth_year          : num [1:4321] 1992 1992 1989 1992 1989 ...
#>  $ age                 : num [1:4321] 18 18 21 18 21 22 18 19 18 21 ...
#>  $ year                : num [1:4321] 2010 2010 2010 2010 2010 2010 2010 2010 2010 2010 ...
#>  $ round               : num [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ overall_pick        : num [1:4321] 1 3 4 8 10 12 15 17 18 20 ...
#>  $ mlb_debut           : Factor w/ 2 levels "no","yes": 2 2 2 2 2 2 1 1 2 1 ...
#>  $ g                   : num [1:4321] 139 222 944 666 788 198 660 224 931 304 ...
#>  $ ab                  : num [1:4321] 486 832 3503 2582 2952 ...
#>  $ r                   : num [1:4321] 83 116 489 483 478 128 301 93 508 148 ...
#>  $ h                   : num [1:4321] 146 224 990 683 822 214 494 185 949 292 ...
#>  $ b2                  : num [1:4321] 30 51 167 115 180 55 92 35 203 62 ...
#>  $ b3                  : num [1:4321] 4 12 12 29 9 0 14 11 35 12 ...
#>  $ hr                  : num [1:4321] 23 23 71 49 126 22 35 18 80 8 ...
#>  $ rbi                 : num [1:4321] 75 116 438 280 513 112 242 105 536 103 ...
#>  $ sb                  : num [1:4321] 29 25 119 294 33 0 73 11 125 34 ...
#>  $ cs                  : num [1:4321] 10 10 56 75 13 2 33 13 40 9 ...
#>  $ bb                  : num [1:4321] 76 97 348 374 325 113 307 109 351 104 ...
#>  $ so                  : num [1:4321] 106 147 400 647 827 165 645 212 821 288 ...
#>  $ sh                  : num [1:4321] 2 1 75 52 0 0 13 0 9 0 ...
#>  $ sf                  : num [1:4321] 2 4 38 20 26 9 17 8 38 11 ...
#>  $ hbp                 : num [1:4321] 3 7 54 44 59 8 17 4 22 15 ...
#>  $ gdp                 : num [1:4321] 10 31 115 28 48 14 29 13 74 29 ...
#>  $ tb                  : num [1:4321] 253 368 1394 1003 1398 ...
#>  $ pa                  : num [1:4321] 569 941 4018 3072 3362 ...
#>  $ xbh                 : num [1:4321] 57 86 250 193 315 77 141 64 318 82 ...
#>  $ avg                 : num [1:4321] 0.3 0.269 0.283 0.265 0.278 0.31 0.226 0.238 0.268 0.258 ...
#>  $ obp                 : num [1:4321] 0.397 0.349 0.353 0.365 0.359 0.408 0.324 0.332 0.334 0.326 ...
#>  $ slg                 : num [1:4321] 0.521 0.442 0.398 0.388 0.474 0.485 0.329 0.381 0.412 0.356 ...
#>  $ ops                 : num [1:4321] 0.917 0.791 0.751 0.753 0.832 0.893 0.653 0.714 0.746 0.682 ...
#>  $ babip               : num [1:4321] 0.343 0.302 0.299 0.333 0.344 0.374 0.302 0.301 0.324 0.336 ...
#>  $ bmi                 : num [1:4321] 26 27.1 24.5 28.1 25.7 ...
#>  $ hr_ab               : num [1:4321] 0.0473 0.0276 0.0203 0.019 0.0427 ...
#>  $ iso                 : num [1:4321] 0.221 0.173 0.115 0.123 0.196 0.175 0.103 0.143 0.144 0.098 ...
#>  $ bb_so               : num [1:4321] 0.717 0.66 0.87 0.578 0.393 ...
#>  $ sbr                 : num [1:4321] 0.744 0.714 0.68 0.797 0.717 ...
#>  $ team_ANA            : int [1:4321] 0 0 0 0 0 0 0 0 1 0 ...
#>  $ team_ARI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_ATL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_BAL            : int [1:4321] 0 1 0 0 0 0 0 0 0 0 ...
#>  $ team_BOS            : int [1:4321] 0 0 0 0 0 0 0 0 0 1 ...
#>  $ team_CHA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CHN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CIN            : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ team_CLE            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_COL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_DET            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_HOU            : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ team_KCA            : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ team_LAN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MON            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_OAK            : int [1:4321] 0 0 0 0 1 0 0 0 0 0 ...
#>  $ team_PHI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_PIT            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SDN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SEA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SFN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SLN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_TBA            : int [1:4321] 0 0 0 0 0 0 0 1 0 0 ...
#>  $ team_TEX            : int [1:4321] 0 0 0 0 0 0 1 0 0 0 ...
#>  $ team_TOR            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_WAS            : int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ schooltype_4Year    : int [1:4321] 0 0 1 0 1 1 0 0 0 1 ...
#>  $ schooltype_HS       : int [1:4321] 0 1 0 1 0 0 1 1 1 0 ...
#>  $ schooltype_JrCollege: int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ bats_B              : int [1:4321] 0 0 0 0 0 1 0 0 1 0 ...
#>  $ bats_L              : int [1:4321] 1 0 0 0 0 0 1 1 0 0 ...
#>  $ bats_R              : int [1:4321] 0 1 1 1 1 0 0 0 0 1 ...
#>  $ throws_L            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ throws_R            : int [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ position_1B         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_2B         : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ position_3B         : int [1:4321] 0 1 0 0 0 0 0 0 1 0 ...
#>  $ position_C          : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ position_CF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_IF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_LF         : int [1:4321] 1 0 0 0 0 0 0 1 0 0 ...
#>  $ position_OF         : int [1:4321] 0 0 0 0 1 0 1 0 0 1 ...
#>  $ position_RF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_SS         : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AB          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AL          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AR          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AZ          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_BC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CA          : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CO          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CT          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CU          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DE          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_FL          : int [1:4321] 0 1 0 0 0 1 0 0 0 0 ...
#>  $ sch_reg_GA          : int [1:4321] 0 0 0 1 0 0 1 0 1 0 ...
#>   [list output truncated]
```

``` r
# Set the seed
set.seed(123)

# Split the data into train and test
df_split_xg <- initial_split(df_mlb_xg, strata = mlb_debut) #strata argument can help the training and test data sets will keep roughly the same proportions of mlb_debut = yes and no as in the original data.
df_train_xg <- training(df_split_xg)
df_test_xg <- testing(df_split_xg)
```

``` r
# create cross-validation resample for tuning the model.
set.seed(123)
dfa_folds_xgb <- vfold_cv(df_train_xg, v = 5, strata = mlb_debut)
```

``` r
# Create a XGBoost model object
# model specification
xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), 
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(), 
  mtry = tune(),
  learn_rate = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  translate()
```

``` r
# set up possible values for these hyperparameters to try
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train_xg),
  learn_rate(),
  size = 5)

xgb_grid
#> # A tibble: 5 × 6
#>   tree_depth min_n loss_reduction sample_size  mtry learn_rate
#>        <int> <int>          <dbl>       <dbl> <int>      <dbl>
#> 1         11    37  0.0351              0.672     8   3.27e- 5
#> 2          8    10  0.00000000885       0.403   192   4.03e- 2
#> 3         13    28  0.000000500         0.139    73   9.39e- 7
#> 4          6     8  0.0000472           0.955   165   6.01e- 8
#> 5          3    24  6.93                0.463   100   2.06e-10
```

#### XGB model with all variables

``` r
# Create recipe 
recipe_xgb_all <- 
  recipe(mlb_debut ~ ., data = df_train_xg) %>% 
  step_zv(all_predictors())


# Create workflow
xgb_workflow_all <- workflow() %>% 
  add_recipe(recipe_xgb_all) %>% 
  add_model(xgb_spec)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
xgb_res_all <- tune_grid(
  xgb_workflow_all,
  resamples = dfa_folds_xgb,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_all <- select_best(xgb_res_all, "roc_auc")
best_auc_all
#> # A tibble: 1 × 7
#>    mtry min_n tree_depth learn_rate loss_reduction sample_size .config          
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>            
#> 1   192    10          8     0.0403  0.00000000885       0.403 Preprocessor1_Mo…
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_xgb_all <- finalize_workflow(
  xgb_workflow_all,
  best_auc_all)
```

``` r
# Fit the tuned model in training data
fit_xgb_all <-fit(final_xgb_all, data = df_train_xg)
#> [13:08:41] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
```

``` r
# Apply model in the testing data
results_xgb_all <- 
  predict(fit_xgb_all, df_test_xg, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_xg, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_xgb_all <- conf_mat(results_xgb_all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_xgb_all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.926
#>  2 kap                  binary         0.731
#>  3 sens                 binary         0.75 
#>  4 spec                 binary         0.962
#>  5 ppv                  binary         0.802
#>  6 npv                  binary         0.949
#>  7 mcc                  binary         0.732
#>  8 j_index              binary         0.712
#>  9 bal_accuracy         binary         0.856
#> 10 detection_prevalence binary         0.159
#> 11 precision            binary         0.802
#> 12 recall               binary         0.75 
#> 13 f_meas               binary         0.775
```

``` r
# AUC value
roc_auc(results_xgb_all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.969
```

``` r
# Plot ROC
roc_curve(results_xgb_all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-61-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_xgb_all %>% 
  ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-62-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_xgb_all, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-63-1.png" width="100%" />

#### XGB model with variables of lasso4all

``` r
# Create recipe 
recipe_xgb_lasso4all <- 
  recipe(mlb_debut ~ . , data = df_train_xg) %>% 
  step_rm(weight, year, round, g, r, b2, bb, pa, obp, slg, hr_ab, iso) %>% 
  step_zv(all_predictors())

# Create workflow
xgb_workflow_lasso4all <- workflow() %>% 
  add_recipe(recipe_xgb_lasso4all) %>% 
  add_model(xgb_spec)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
xgb_res_lasso4all <- tune_grid(
  xgb_workflow_lasso4all,
  resamples = dfa_folds_xgb,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_lasso4all <- select_best(xgb_res_lasso4all, "roc_auc")
best_auc_lasso4all
#> # A tibble: 1 × 7
#>    mtry min_n tree_depth learn_rate loss_reduction sample_size .config          
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>            
#> 1   192    10          8     0.0403  0.00000000885       0.403 Preprocessor1_Mo…
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_xgb_lasso4all <- finalize_workflow(
  xgb_workflow_lasso4all,
  best_auc_lasso4all
)
```

``` r
# Fit the tuned model in training data
fit_xgb_lasso4all <-fit(final_xgb_lasso4all, data = df_train_xg)
#> [13:11:48] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
```

``` r
# Apply model in the testing data
results_xgb_lasso4all <- 
  predict(fit_xgb_lasso4all, df_test_xg, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_xg, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_xgb_lasso4all <- conf_mat(results_xgb_lasso4all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_xgb_lasso4all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.927
#>  2 kap                  binary         0.731
#>  3 sens                 binary         0.739
#>  4 spec                 binary         0.965
#>  5 ppv                  binary         0.814
#>  6 npv                  binary         0.947
#>  7 mcc                  binary         0.733
#>  8 j_index              binary         0.705
#>  9 bal_accuracy         binary         0.852
#> 10 detection_prevalence binary         0.154
#> 11 precision            binary         0.814
#> 12 recall               binary         0.739
#> 13 f_meas               binary         0.775
```

``` r
# AUC value
roc_auc(results_xgb_lasso4all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.970
```

``` r
# Plot ROC
roc_curve(results_xgb_lasso4all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-72-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_xgb_lasso4all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-73-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_xgb_lasso4all, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-74-1.png" width="100%" />

#### XGBoost model with expert-selected variables

``` r
# Create recipe 
recipe_xgb_selected_lch <- 
  recipe(mlb_debut ~  bats_B + bats_L + bats_R + age + bmi  + round + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + obp + slg + ops + iso + bb_so  + hr_ab +sbr + ab, data = df_train_xg) %>% 
  step_zv(all_predictors())


# Create workflow
xgb_workflow_selected_lch <- workflow() %>% 
  add_recipe(recipe_xgb_selected_lch) %>% 
  add_model(xgb_spec)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
xgb_res_selected_lch <- tune_grid(
  xgb_workflow_selected_lch,
  resamples = dfa_folds_xgb,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_selected_lch <- select_best(xgb_res_selected_lch, "roc_auc")
best_auc_selected_lch
#> # A tibble: 1 × 7
#>    mtry min_n tree_depth learn_rate loss_reduction sample_size .config          
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>            
#> 1   192    10          8     0.0403  0.00000000885       0.403 Preprocessor1_Mo…
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_xgb_selected_lch <- finalize_workflow(
  xgb_workflow_selected_lch,
  best_auc_selected_lch
)
```

``` r
# Fit the tuned model in training data
fit_xgb_selected_lch <-fit(final_xgb_selected_lch, data = df_train_xg)
#> [13:13:22] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
```

``` r
# Apply model in the testing data
results_xgb_selected_lch <- 
  predict(fit_xgb_selected_lch, df_test_xg, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_xg, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_xgb_selected_lch <- conf_mat(results_xgb_selected_lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_xgb_selected_lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.917
#>  2 kap                  binary         0.708
#>  3 sens                 binary         0.766
#>  4 spec                 binary         0.948
#>  5 ppv                  binary         0.75 
#>  6 npv                  binary         0.952
#>  7 mcc                  binary         0.708
#>  8 j_index              binary         0.714
#>  9 bal_accuracy         binary         0.857
#> 10 detection_prevalence binary         0.174
#> 11 precision            binary         0.75 
#> 12 recall               binary         0.766
#> 13 f_meas               binary         0.758
```

``` r
# AUC value
roc_auc(results_xgb_selected_lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.963
```

``` r
# Plot ROC
roc_curve(results_xgb_selected_lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-83-1.png" width="100%" />

``` r
#Plot predicted probability vs OPS
results_xgb_selected_lch %>% 
  ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-84-1.png" width="100%" />

``` r
#Find out top10 most important variables in this model
last_fit(final_xgb_selected_lch, split = df_split_xg) %>%   
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-85-1.png" width="100%" />

#### XGB model with variables of lasso4lch

``` r
# create recipe 
recipe_xgb_lasso4lch <- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + slg + ops + bb_so +sbr + ab, data = df_train_xg) %>% 
  step_zv(all_predictors())


# create workflow
xgb_workflow_lasso4lch <- workflow() %>% 
  add_recipe(recipe_xgb_lasso4lch) %>% 
  add_model(xgb_spec)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
xgb_res_lasso4lch <- tune_grid(
  xgb_workflow_lasso4lch,
  resamples = dfa_folds_xgb,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_lasso4lch <- select_best(xgb_res_lasso4lch, "roc_auc")
best_auc_lasso4lch
#> # A tibble: 1 × 7
#>    mtry min_n tree_depth learn_rate loss_reduction sample_size .config          
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>            
#> 1   192    10          8     0.0403  0.00000000885       0.403 Preprocessor1_Mo…
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_xgb_lasso4lch <- finalize_workflow(
  xgb_workflow_lasso4lch,
  best_auc_lasso4lch
)
```

``` r
#Fit the tuned model in training data
fit_xgb_lasso4lch <-fit(final_xgb_lasso4lch, data = df_train_xg)
#> [13:14:21] WARNING: amalgamation/../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
```

``` r
# Apply model in the testing data
results_xgb_lasso4lch <- 
  predict(fit_xgb_lasso4lch, df_test_xg, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_xg, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_xgb_lasso4lch <- conf_mat(results_xgb_lasso4lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_xgb_lasso4lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.917
#>  2 kap                  binary         0.705
#>  3 sens                 binary         0.755
#>  4 spec                 binary         0.950
#>  5 ppv                  binary         0.755
#>  6 npv                  binary         0.950
#>  7 mcc                  binary         0.705
#>  8 j_index              binary         0.705
#>  9 bal_accuracy         binary         0.853
#> 10 detection_prevalence binary         0.170
#> 11 precision            binary         0.755
#> 12 recall               binary         0.755
#> 13 f_meas               binary         0.755
```

``` r
# AUC value
roc_auc(results_xgb_lasso4lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.966
```

``` r
# Plot ROC
roc_curve(results_xgb_lasso4lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-94-1.png" width="100%" />

``` r
#Plot predicted probability vs OPS
results_xgb_lasso4lch %>% 
  ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-95-1.png" width="100%" />

``` r
#Find out top10 most important variables in this model
last_fit(fit_xgb_lasso4lch, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-96-1.png" width="100%" />

## Random Forest

``` r
# Prepare dataset for Random Forest
df_mlb_ranfor <- df_mlb %>% 
  select(-name, -highLevel, -school) %>% # Remove variables of name, high-level and school
  fastDummies::dummy_cols(select_columns = c( 'team', 'schooltype', 'bats', 'throws', 'position', 'sch_reg', 'birth_place'), remove_selected_columns = TRUE) # Converting the categorical variable into dummies

str(df_mlb_ranfor)
#> tibble [4,321 × 222] (S3: tbl_df/tbl/data.frame)
#>  $ height              : num [1:4321] 191 191 183 175 183 188 185 183 191 191 ...
#>  $ weight              : num [1:4321] 95 99 82 86 86 104 86 98 86 88 ...
#>  $ birth_year          : num [1:4321] 1992 1992 1989 1992 1989 ...
#>  $ age                 : num [1:4321] 18 18 21 18 21 22 18 19 18 21 ...
#>  $ year                : num [1:4321] 2010 2010 2010 2010 2010 2010 2010 2010 2010 2010 ...
#>  $ round               : num [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ overall_pick        : num [1:4321] 1 3 4 8 10 12 15 17 18 20 ...
#>  $ mlb_debut           : Factor w/ 2 levels "no","yes": 2 2 2 2 2 2 1 1 2 1 ...
#>  $ g                   : num [1:4321] 139 222 944 666 788 198 660 224 931 304 ...
#>  $ ab                  : num [1:4321] 486 832 3503 2582 2952 ...
#>  $ r                   : num [1:4321] 83 116 489 483 478 128 301 93 508 148 ...
#>  $ h                   : num [1:4321] 146 224 990 683 822 214 494 185 949 292 ...
#>  $ b2                  : num [1:4321] 30 51 167 115 180 55 92 35 203 62 ...
#>  $ b3                  : num [1:4321] 4 12 12 29 9 0 14 11 35 12 ...
#>  $ hr                  : num [1:4321] 23 23 71 49 126 22 35 18 80 8 ...
#>  $ rbi                 : num [1:4321] 75 116 438 280 513 112 242 105 536 103 ...
#>  $ sb                  : num [1:4321] 29 25 119 294 33 0 73 11 125 34 ...
#>  $ cs                  : num [1:4321] 10 10 56 75 13 2 33 13 40 9 ...
#>  $ bb                  : num [1:4321] 76 97 348 374 325 113 307 109 351 104 ...
#>  $ so                  : num [1:4321] 106 147 400 647 827 165 645 212 821 288 ...
#>  $ sh                  : num [1:4321] 2 1 75 52 0 0 13 0 9 0 ...
#>  $ sf                  : num [1:4321] 2 4 38 20 26 9 17 8 38 11 ...
#>  $ hbp                 : num [1:4321] 3 7 54 44 59 8 17 4 22 15 ...
#>  $ gdp                 : num [1:4321] 10 31 115 28 48 14 29 13 74 29 ...
#>  $ tb                  : num [1:4321] 253 368 1394 1003 1398 ...
#>  $ pa                  : num [1:4321] 569 941 4018 3072 3362 ...
#>  $ xbh                 : num [1:4321] 57 86 250 193 315 77 141 64 318 82 ...
#>  $ avg                 : num [1:4321] 0.3 0.269 0.283 0.265 0.278 0.31 0.226 0.238 0.268 0.258 ...
#>  $ obp                 : num [1:4321] 0.397 0.349 0.353 0.365 0.359 0.408 0.324 0.332 0.334 0.326 ...
#>  $ slg                 : num [1:4321] 0.521 0.442 0.398 0.388 0.474 0.485 0.329 0.381 0.412 0.356 ...
#>  $ ops                 : num [1:4321] 0.917 0.791 0.751 0.753 0.832 0.893 0.653 0.714 0.746 0.682 ...
#>  $ babip               : num [1:4321] 0.343 0.302 0.299 0.333 0.344 0.374 0.302 0.301 0.324 0.336 ...
#>  $ bmi                 : num [1:4321] 26 27.1 24.5 28.1 25.7 ...
#>  $ hr_ab               : num [1:4321] 0.0473 0.0276 0.0203 0.019 0.0427 ...
#>  $ iso                 : num [1:4321] 0.221 0.173 0.115 0.123 0.196 0.175 0.103 0.143 0.144 0.098 ...
#>  $ bb_so               : num [1:4321] 0.717 0.66 0.87 0.578 0.393 ...
#>  $ sbr                 : num [1:4321] 0.744 0.714 0.68 0.797 0.717 ...
#>  $ team_ANA            : int [1:4321] 0 0 0 0 0 0 0 0 1 0 ...
#>  $ team_ARI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_ATL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_BAL            : int [1:4321] 0 1 0 0 0 0 0 0 0 0 ...
#>  $ team_BOS            : int [1:4321] 0 0 0 0 0 0 0 0 0 1 ...
#>  $ team_CHA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CHN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CIN            : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ team_CLE            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_COL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_DET            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_HOU            : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ team_KCA            : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ team_LAN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MON            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_OAK            : int [1:4321] 0 0 0 0 1 0 0 0 0 0 ...
#>  $ team_PHI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_PIT            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SDN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SEA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SFN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SLN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_TBA            : int [1:4321] 0 0 0 0 0 0 0 1 0 0 ...
#>  $ team_TEX            : int [1:4321] 0 0 0 0 0 0 1 0 0 0 ...
#>  $ team_TOR            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_WAS            : int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ schooltype_4Year    : int [1:4321] 0 0 1 0 1 1 0 0 0 1 ...
#>  $ schooltype_HS       : int [1:4321] 0 1 0 1 0 0 1 1 1 0 ...
#>  $ schooltype_JrCollege: int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ bats_B              : int [1:4321] 0 0 0 0 0 1 0 0 1 0 ...
#>  $ bats_L              : int [1:4321] 1 0 0 0 0 0 1 1 0 0 ...
#>  $ bats_R              : int [1:4321] 0 1 1 1 1 0 0 0 0 1 ...
#>  $ throws_L            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ throws_R            : int [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ position_1B         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_2B         : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ position_3B         : int [1:4321] 0 1 0 0 0 0 0 0 1 0 ...
#>  $ position_C          : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ position_CF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_IF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_LF         : int [1:4321] 1 0 0 0 0 0 0 1 0 0 ...
#>  $ position_OF         : int [1:4321] 0 0 0 0 1 0 1 0 0 1 ...
#>  $ position_RF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_SS         : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AB          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AL          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AR          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AZ          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_BC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CA          : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CO          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CT          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CU          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DE          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_FL          : int [1:4321] 0 1 0 0 0 1 0 0 0 0 ...
#>  $ sch_reg_GA          : int [1:4321] 0 0 0 1 0 0 1 0 1 0 ...
#>   [list output truncated]
```

``` r
# Set the seed
set.seed(123)

# Split the data into train and test
df_split_ranfor <- initial_split(df_mlb_ranfor, strata = mlb_debut) #strata argument can help the training and test data sets will keep roughly the same proportions of mlb_debut = yes and no as in the original data.
df_train_ranfor <- training(df_split_ranfor)
df_test_ranfor <- testing(df_split_ranfor)
```

``` r
# create cross-validation resample for tuning the model.
set.seed(123)
dfa_folds_ranfor <- vfold_cv(df_train_ranfor, v = 5, strata = mlb_debut)
```

``` r
# Create a random forest model object
ranfor <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")  
```

``` r
# Set up possible values for these hyperparameters to try
ranfor_grid <- grid_regular(
  min_n(),
  finalize(mtry(), df_train_ranfor),
  levels = 5)

ranfor_grid
#> # A tibble: 25 × 2
#>    min_n  mtry
#>    <int> <int>
#>  1     2     1
#>  2    11     1
#>  3    21     1
#>  4    30     1
#>  5    40     1
#>  6     2    56
#>  7    11    56
#>  8    21    56
#>  9    30    56
#> 10    40    56
#> # … with 15 more rows
```

#### Random Forest model with all variables

``` r
# Create recipe
recipe_ranfor_all <- 
  recipe(mlb_debut ~ ., data = df_train_ranfor) %>% 
  step_zv(all_predictors())

# Create workflow
ranfor_workflow_all <- workflow() %>% 
  add_recipe(recipe_ranfor_all) %>% 
  add_model(ranfor)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
ranfor_res_all <- tune_grid(
  ranfor_workflow_all,
  resamples = dfa_folds_ranfor,
  grid = ranfor_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
ranfor_best_auc_all <- select_best(ranfor_res_all, "roc_auc")
ranfor_best_auc_all
#> # A tibble: 1 × 3
#>    mtry min_n .config              
#>   <int> <int> <chr>                
#> 1   111     2 Preprocessor1_Model11
```

``` r
# Finalize our tuneable workflow with these parameter values.
final_ranfor_all <- finalize_workflow(
  ranfor_workflow_all,
  ranfor_best_auc_all
)
```

``` r
# Fit the tuned model in training data
fit_ranfor_all <-fit(final_ranfor_all, data = df_train_ranfor)
```

``` r
# Apply model in the testing data
results_ranfor_all <- 
  predict(fit_ranfor_all, df_test_ranfor, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_ranfor, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_ranfor_all <- conf_mat(results_ranfor_all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_ranfor_all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.926
#>  2 kap                  binary         0.736
#>  3 sens                 binary         0.772
#>  4 spec                 binary         0.958
#>  5 ppv                  binary         0.789
#>  6 npv                  binary         0.953
#>  7 mcc                  binary         0.736
#>  8 j_index              binary         0.729
#>  9 bal_accuracy         binary         0.865
#> 10 detection_prevalence binary         0.167
#> 11 precision            binary         0.789
#> 12 recall               binary         0.772
#> 13 f_meas               binary         0.780
```

``` r
# AUC value
roc_auc(results_ranfor_all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.961
```

``` r
# Plot ROC
roc_curve(results_ranfor_all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-110-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_ranfor_all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-111-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_ranfor_all, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-112-1.png" width="100%" />

#### Random Forest model with variables of lasso4all

``` r
# Create recipe
recipe_ranfor_lasso4all <- 
  recipe(mlb_debut ~ ., data = df_train_ranfor) %>% 
  step_rm(weight, year, round, g, r, b2, bb, pa, obp, slg, hr_ab, iso)  %>% 
  step_zv(all_predictors())

# Create workflow
ranfor_workflow_lasso4all <- workflow() %>% 
  add_recipe(recipe_ranfor_lasso4all) %>% 
  add_model(ranfor)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
ranfor_res_lasso4all <- tune_grid(
  ranfor_workflow_lasso4all,
  resamples = dfa_folds_ranfor,
  grid = ranfor_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
ranfor_best_auc_lasso4all <- select_best(ranfor_res_lasso4all, "roc_auc")
ranfor_best_auc_lasso4all
#> # A tibble: 1 × 3
#>    mtry min_n .config              
#>   <int> <int> <chr>                
#> 1   111    11 Preprocessor1_Model12
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_ranfor_lasso4all <- finalize_workflow(
  ranfor_workflow_lasso4all,
  ranfor_best_auc_lasso4all)
```

``` r
# Fit the tuned model in training data
fit_ranfor_lasso4all <-fit(final_ranfor_lasso4all, data = df_train_ranfor)
```

``` r
# Apply model in the testing data
results_ranfor_lasso4all <- 
  predict(fit_ranfor_lasso4all, df_test_ranfor, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_ranfor, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_ranfor_lasso4all <- conf_mat(results_ranfor_lasso4all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_ranfor_lasso4all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.928
#>  2 kap                  binary         0.741
#>  3 sens                 binary         0.772
#>  4 spec                 binary         0.960
#>  5 ppv                  binary         0.798
#>  6 npv                  binary         0.953
#>  7 mcc                  binary         0.741
#>  8 j_index              binary         0.732
#>  9 bal_accuracy         binary         0.866
#> 10 detection_prevalence binary         0.165
#> 11 precision            binary         0.798
#> 12 recall               binary         0.772
#> 13 f_meas               binary         0.785
```

``` r
# AUC value
roc_auc(results_ranfor_lasso4all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.962
```

``` r
# Plot ROC
roc_curve(results_ranfor_lasso4all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-121-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_ranfor_lasso4all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-122-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_ranfor_lasso4all, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-123-1.png" width="100%" />

#### Random Forest model with expert-selected variables

``` r
# Create recipe
recipe_ranfor_selected_lch <- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi  + round + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + obp + slg + ops + iso + bb_so  + hr_ab +sbr + ab, data = df_train_ranfor) %>%
  step_zv(all_predictors())

# Create workflow
ranfor_workflow_selected_lch <- workflow() %>% 
  add_recipe(recipe_ranfor_selected_lch) %>% 
  add_model(ranfor)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
ranfor_res_selected_lch <- tune_grid(
  ranfor_workflow_selected_lch,
  resamples = dfa_folds_ranfor,
  grid = ranfor_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
ranfor_best_auc_selected_lch <- select_best(ranfor_res_selected_lch, "roc_auc")
ranfor_best_auc_selected_lch
#> # A tibble: 1 × 3
#>    mtry min_n .config              
#>   <int> <int> <chr>                
#> 1   166    40 Preprocessor1_Model20
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_ranfor_selected_lch <- finalize_workflow(
  ranfor_workflow_selected_lch,
  ranfor_best_auc_selected_lch
)
```

``` r
# Fit the tuned model in training data
fit_ranfor_selected_lch <-fit(final_ranfor_selected_lch, data = df_train_ranfor)
```

``` r
# Apply model in the testing data
results_ranfor_selected_lch <- 
  predict(fit_ranfor_selected_lch, df_test_ranfor, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_ranfor, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_ranfor_selected_lch <- conf_mat(results_ranfor_selected_lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_ranfor_selected_lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.920
#>  2 kap                  binary         0.717
#>  3 sens                 binary         0.772
#>  4 spec                 binary         0.950
#>  5 ppv                  binary         0.759
#>  6 npv                  binary         0.953
#>  7 mcc                  binary         0.717
#>  8 j_index              binary         0.722
#>  9 bal_accuracy         binary         0.861
#> 10 detection_prevalence binary         0.173
#> 11 precision            binary         0.759
#> 12 recall               binary         0.772
#> 13 f_meas               binary         0.765
```

``` r
# AUC value
roc_auc(results_ranfor_selected_lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.957
```

``` r
# Plot ROC
roc_curve(results_ranfor_selected_lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-132-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_ranfor_selected_lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-133-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_ranfor_selected_lch, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-134-1.png" width="100%" />

#### Random Forest model with variables of Lasso4lch

``` r
# Create recipe
recipe_ranfor_lasso4lch <- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + slg + ops + bb_so +sbr + ab, data = df_train_ranfor) %>% 
  step_zv(all_predictors())

# Create workflow
ranfor_workflow_lasso4lch <- workflow() %>% 
  add_recipe(recipe_ranfor_lasso4lch) %>% 
  add_model(ranfor)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

dfa_folds_ranfor <- vfold_cv(df_train_ranfor)

set.seed(123)
ranfor_res_lasso4lch <- tune_grid(
  ranfor_workflow_lasso4lch,
  resamples = dfa_folds_ranfor,
  grid = ranfor_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
ranfor_best_auc_lasso4lch <- select_best(ranfor_res_lasso4lch, "roc_auc")
ranfor_best_auc_lasso4lch
#> # A tibble: 1 × 3
#>    mtry min_n .config              
#>   <int> <int> <chr>                
#> 1   111    11 Preprocessor1_Model12
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_ranfor_lasso4lch <- finalize_workflow(
  ranfor_workflow_lasso4lch,
  ranfor_best_auc_lasso4lch
)
```

``` r
# Fit the tuned model in training data
fit_ranfor_lasso4lch <-fit(final_ranfor_lasso4lch, data = df_train_ranfor)
```

``` r
# Apply model in the testing data
results_ranfor_lasso4lch <- 
  predict(fit_ranfor_lasso4lch, df_test_ranfor, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_ranfor, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_ranfor_lasso4lch <- conf_mat(results_ranfor_lasso4lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_ranfor_lasso4lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.920
#>  2 kap                  binary         0.718
#>  3 sens                 binary         0.766
#>  4 spec                 binary         0.952
#>  5 ppv                  binary         0.766
#>  6 npv                  binary         0.952
#>  7 mcc                  binary         0.718
#>  8 j_index              binary         0.718
#>  9 bal_accuracy         binary         0.859
#> 10 detection_prevalence binary         0.170
#> 11 precision            binary         0.766
#> 12 recall               binary         0.766
#> 13 f_meas               binary         0.766
```

``` r
# AUC value
roc_auc(results_ranfor_lasso4lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.955
```

``` r
# Plot ROC
roc_curve(results_ranfor_lasso4lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-143-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_ranfor_lasso4lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-144-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_ranfor_lasso4lch, split = df_split_xg) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-145-1.png" width="100%" />

#### Decision Tree

``` r
# Prepare dataset for Decision Tree
df_mlb_dt <- df_mlb %>% 
  select(-name, -highLevel, -school) %>%  # Remove variables of name, high-level and school
  fastDummies::dummy_cols(select_columns = c( 'team', 'schooltype', 'bats', 'throws', 'position', 'sch_reg', 'birth_place'), remove_selected_columns = TRUE) # Converting the categorical variable into dummies

str(df_mlb_dt)
#> tibble [4,321 × 222] (S3: tbl_df/tbl/data.frame)
#>  $ height              : num [1:4321] 191 191 183 175 183 188 185 183 191 191 ...
#>  $ weight              : num [1:4321] 95 99 82 86 86 104 86 98 86 88 ...
#>  $ birth_year          : num [1:4321] 1992 1992 1989 1992 1989 ...
#>  $ age                 : num [1:4321] 18 18 21 18 21 22 18 19 18 21 ...
#>  $ year                : num [1:4321] 2010 2010 2010 2010 2010 2010 2010 2010 2010 2010 ...
#>  $ round               : num [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ overall_pick        : num [1:4321] 1 3 4 8 10 12 15 17 18 20 ...
#>  $ mlb_debut           : Factor w/ 2 levels "no","yes": 2 2 2 2 2 2 1 1 2 1 ...
#>  $ g                   : num [1:4321] 139 222 944 666 788 198 660 224 931 304 ...
#>  $ ab                  : num [1:4321] 486 832 3503 2582 2952 ...
#>  $ r                   : num [1:4321] 83 116 489 483 478 128 301 93 508 148 ...
#>  $ h                   : num [1:4321] 146 224 990 683 822 214 494 185 949 292 ...
#>  $ b2                  : num [1:4321] 30 51 167 115 180 55 92 35 203 62 ...
#>  $ b3                  : num [1:4321] 4 12 12 29 9 0 14 11 35 12 ...
#>  $ hr                  : num [1:4321] 23 23 71 49 126 22 35 18 80 8 ...
#>  $ rbi                 : num [1:4321] 75 116 438 280 513 112 242 105 536 103 ...
#>  $ sb                  : num [1:4321] 29 25 119 294 33 0 73 11 125 34 ...
#>  $ cs                  : num [1:4321] 10 10 56 75 13 2 33 13 40 9 ...
#>  $ bb                  : num [1:4321] 76 97 348 374 325 113 307 109 351 104 ...
#>  $ so                  : num [1:4321] 106 147 400 647 827 165 645 212 821 288 ...
#>  $ sh                  : num [1:4321] 2 1 75 52 0 0 13 0 9 0 ...
#>  $ sf                  : num [1:4321] 2 4 38 20 26 9 17 8 38 11 ...
#>  $ hbp                 : num [1:4321] 3 7 54 44 59 8 17 4 22 15 ...
#>  $ gdp                 : num [1:4321] 10 31 115 28 48 14 29 13 74 29 ...
#>  $ tb                  : num [1:4321] 253 368 1394 1003 1398 ...
#>  $ pa                  : num [1:4321] 569 941 4018 3072 3362 ...
#>  $ xbh                 : num [1:4321] 57 86 250 193 315 77 141 64 318 82 ...
#>  $ avg                 : num [1:4321] 0.3 0.269 0.283 0.265 0.278 0.31 0.226 0.238 0.268 0.258 ...
#>  $ obp                 : num [1:4321] 0.397 0.349 0.353 0.365 0.359 0.408 0.324 0.332 0.334 0.326 ...
#>  $ slg                 : num [1:4321] 0.521 0.442 0.398 0.388 0.474 0.485 0.329 0.381 0.412 0.356 ...
#>  $ ops                 : num [1:4321] 0.917 0.791 0.751 0.753 0.832 0.893 0.653 0.714 0.746 0.682 ...
#>  $ babip               : num [1:4321] 0.343 0.302 0.299 0.333 0.344 0.374 0.302 0.301 0.324 0.336 ...
#>  $ bmi                 : num [1:4321] 26 27.1 24.5 28.1 25.7 ...
#>  $ hr_ab               : num [1:4321] 0.0473 0.0276 0.0203 0.019 0.0427 ...
#>  $ iso                 : num [1:4321] 0.221 0.173 0.115 0.123 0.196 0.175 0.103 0.143 0.144 0.098 ...
#>  $ bb_so               : num [1:4321] 0.717 0.66 0.87 0.578 0.393 ...
#>  $ sbr                 : num [1:4321] 0.744 0.714 0.68 0.797 0.717 ...
#>  $ team_ANA            : int [1:4321] 0 0 0 0 0 0 0 0 1 0 ...
#>  $ team_ARI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_ATL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_BAL            : int [1:4321] 0 1 0 0 0 0 0 0 0 0 ...
#>  $ team_BOS            : int [1:4321] 0 0 0 0 0 0 0 0 0 1 ...
#>  $ team_CHA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CHN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CIN            : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ team_CLE            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_COL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_DET            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_HOU            : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ team_KCA            : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ team_LAN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MON            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_OAK            : int [1:4321] 0 0 0 0 1 0 0 0 0 0 ...
#>  $ team_PHI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_PIT            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SDN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SEA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SFN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SLN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_TBA            : int [1:4321] 0 0 0 0 0 0 0 1 0 0 ...
#>  $ team_TEX            : int [1:4321] 0 0 0 0 0 0 1 0 0 0 ...
#>  $ team_TOR            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_WAS            : int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ schooltype_4Year    : int [1:4321] 0 0 1 0 1 1 0 0 0 1 ...
#>  $ schooltype_HS       : int [1:4321] 0 1 0 1 0 0 1 1 1 0 ...
#>  $ schooltype_JrCollege: int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ bats_B              : int [1:4321] 0 0 0 0 0 1 0 0 1 0 ...
#>  $ bats_L              : int [1:4321] 1 0 0 0 0 0 1 1 0 0 ...
#>  $ bats_R              : int [1:4321] 0 1 1 1 1 0 0 0 0 1 ...
#>  $ throws_L            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ throws_R            : int [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ position_1B         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_2B         : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ position_3B         : int [1:4321] 0 1 0 0 0 0 0 0 1 0 ...
#>  $ position_C          : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ position_CF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_IF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_LF         : int [1:4321] 1 0 0 0 0 0 0 1 0 0 ...
#>  $ position_OF         : int [1:4321] 0 0 0 0 1 0 1 0 0 1 ...
#>  $ position_RF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_SS         : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AB          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AL          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AR          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AZ          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_BC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CA          : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CO          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CT          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CU          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DE          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_FL          : int [1:4321] 0 1 0 0 0 1 0 0 0 0 ...
#>  $ sch_reg_GA          : int [1:4321] 0 0 0 1 0 0 1 0 1 0 ...
#>   [list output truncated]
```

``` r
# Set the seed
set.seed(123)

# Split the data into train and test
df_split_dt <- initial_split(df_mlb_dt, strata = mlb_debut) #strata argument can help the training and test data sets will keep roughly the same proportions of mlb_debut = yes and no as in the original data.
df_train_dt <- training(df_split_dt)
df_test_dt <- testing(df_split_dt)
```

``` r
# create cross-validation resample for tuning the model.
set.seed(123)
dfa_folds_dt <- vfold_cv(df_train_dt, v = 5, strata = mlb_debut)
```

``` r
# Create a Decision Tree model object
# model specification
dt <- decision_tree(
  tree_depth = tune(),
  min_n = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification") %>% 
  translate()
```

``` r
# Set up possible values for these hyperparameters to try
dt_grid <- grid_regular(
  tree_depth(),
  min_n(),
  levels = 5)

dt_grid
#> # A tibble: 25 × 2
#>    tree_depth min_n
#>         <int> <int>
#>  1          1     2
#>  2          4     2
#>  3          8     2
#>  4         11     2
#>  5         15     2
#>  6          1    11
#>  7          4    11
#>  8          8    11
#>  9         11    11
#> 10         15    11
#> # … with 15 more rows
```

#### Decision Tree model with all variables

``` r
# Create recipe 
recipe_dt_all <- 
  recipe(mlb_debut ~ ., data = df_train_dt) %>% 
  step_zv(all_predictors())

# Create workflow
dt_workflow_all <- workflow() %>% 
  add_recipe(recipe_dt_all) %>% 
  add_model(dt)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
dt_res_all <- tune_grid(
  dt_workflow_all,
  resamples = dfa_folds_dt,
  grid = dt_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_all <- select_best(dt_res_all, "roc_auc")
best_auc_all
#> # A tibble: 1 × 3
#>   tree_depth min_n .config              
#>        <int> <int> <chr>                
#> 1          8     2 Preprocessor1_Model03
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_dt_all <- finalize_workflow(
  dt_workflow_all,
  best_auc_all
)
```

``` r
# Fit the tuned model in training data
fit_dt_all <-fit(final_dt_all, data = df_train_dt)
```

``` r
# Apply model in the testing data
results_dt_all <- 
  predict(fit_dt_all, df_test_dt, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_dt, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_dt_all <- conf_mat(results_dt_all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_dt_all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.914
#>  2 kap                  binary         0.682
#>  3 sens                 binary         0.696
#>  4 spec                 binary         0.959
#>  5 ppv                  binary         0.776
#>  6 npv                  binary         0.939
#>  7 mcc                  binary         0.684
#>  8 j_index              binary         0.654
#>  9 bal_accuracy         binary         0.827
#> 10 detection_prevalence binary         0.153
#> 11 precision            binary         0.776
#> 12 recall               binary         0.696
#> 13 f_meas               binary         0.734
```

``` r
# AUC value
roc_auc(results_dt_all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.899
```

``` r
# Plot ROC
roc_curve(results_dt_all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-159-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_dt_all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-160-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_dt_all, split = df_split_dt) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-161-1.png" width="100%" />

#### Decision Tree model with variables of lasso4all

``` r
# Create recipe 
recipe_dt_lasso4all <- 
  recipe(mlb_debut ~ ., data = df_train_dt) %>% 
  step_rm(weight, year, round, g, r, b2, bb, pa, obp, slg, hr_ab, iso) %>% 
  step_zv(all_predictors())

# Create workflow
dt_workflow_lasso4all <- workflow() %>% 
  add_recipe(recipe_dt_lasso4all) %>% 
  add_model(dt)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
dt_res_lasso4all <- tune_grid(
  dt_workflow_lasso4all,
  resamples = dfa_folds_dt,
  grid = dt_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_lasso4all <- select_best(dt_res_lasso4all, "roc_auc")
best_auc_lasso4all
#> # A tibble: 1 × 3
#>   tree_depth min_n .config              
#>        <int> <int> <chr>                
#> 1          8     2 Preprocessor1_Model03
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_dt_lasso4all <- finalize_workflow(
  dt_workflow_lasso4all,
  best_auc_lasso4all)
```

``` r
# Fit the tuned model in training data
fit_dt_lasso4all <-fit(final_dt_lasso4all, data = df_train_dt)
```

``` r
# Apply model in the testing data
results_dt_lasso4all <- 
  predict(fit_dt_lasso4all, df_test_dt, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_dt, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_dt_lasso4all <- conf_mat(results_dt_lasso4all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_dt_lasso4all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.913
#>  2 kap                  binary         0.684
#>  3 sens                 binary         0.712
#>  4 spec                 binary         0.954
#>  5 ppv                  binary         0.762
#>  6 npv                  binary         0.942
#>  7 mcc                  binary         0.685
#>  8 j_index              binary         0.666
#>  9 bal_accuracy         binary         0.833
#> 10 detection_prevalence binary         0.159
#> 11 precision            binary         0.762
#> 12 recall               binary         0.712
#> 13 f_meas               binary         0.736
```

``` r
# AUC value
roc_auc(results_dt_lasso4all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.895
```

``` r
# Plot ROC
roc_curve(results_dt_lasso4all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-170-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_dt_lasso4all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-171-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_dt_lasso4all, split = df_split_dt) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-172-1.png" width="100%" />

#### Decision Tree model with expert-selected variables

``` r
# Create recipe 
recipe_dt_lch <- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi  + round + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + obp + slg + ops + iso + bb_so  + hr_ab +sbr + ab, data = df_train_dt) %>% 
  step_zv(all_predictors())

# Create workflow
dt_workflow_lch <- workflow() %>% 
  add_recipe(recipe_dt_lch) %>% 
  add_model(dt)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
dt_res_lch <- tune_grid(
  dt_workflow_lch,
  resamples = dfa_folds_dt,
  grid = dt_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_lch <- select_best(dt_res_lch, "roc_auc")
best_auc_lch
#> # A tibble: 1 × 3
#>   tree_depth min_n .config              
#>        <int> <int> <chr>                
#> 1          8     2 Preprocessor1_Model03
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_dt_lch <- finalize_workflow(
  dt_workflow_lch,
  best_auc_lch)
```

``` r
# Fit the tuned model in training data
fit_dt_lch <-fit(final_dt_lch, data = df_train_dt)
```

``` r
# Apply model in the testing data
results_dt_lch <- 
  predict(fit_dt_lch, df_test_dt, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_dt, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_dt_lch <- conf_mat(results_dt_lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_dt_lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.904
#>  2 kap                  binary         0.627
#>  3 sens                 binary         0.609
#>  4 spec                 binary         0.964
#>  5 ppv                  binary         0.778
#>  6 npv                  binary         0.923
#>  7 mcc                  binary         0.634
#>  8 j_index              binary         0.573
#>  9 bal_accuracy         binary         0.787
#> 10 detection_prevalence binary         0.133
#> 11 precision            binary         0.778
#> 12 recall               binary         0.609
#> 13 f_meas               binary         0.683
```

``` r
# AUC value
roc_auc(results_dt_lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.886
```

``` r
# Plot ROC
roc_curve(results_dt_lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-181-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_dt_lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-182-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_dt_lch, split = df_split_dt) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-183-1.png" width="100%" />

#### Decision Tree model with variables of lasso4lch

``` r
# Create recipe 
recipe_dt_lasso4lch <- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + slg + ops + bb_so +sbr + ab, data = df_train_dt) %>% 
  step_zv(all_predictors())

# Create workflow
dt_workflow_lasso4lch <- workflow() %>% 
  add_recipe(recipe_dt_lasso4lch) %>% 
  add_model(dt)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
dt_res_lasso4lch <- tune_grid(
  dt_workflow_lasso4lch,
  resamples = dfa_folds_dt,
  grid = dt_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_auc_lasso4lch <- select_best(dt_res_lasso4lch, "roc_auc")
best_auc_lasso4lch
#> # A tibble: 1 × 3
#>   tree_depth min_n .config              
#>        <int> <int> <chr>                
#> 1          4     2 Preprocessor1_Model02
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_dt_lasso4lch <- finalize_workflow(
  dt_workflow_lasso4lch,
  best_auc_lasso4lch)
```

``` r
# Fit the tuned model in training data
fit_dt_lasso4lch <-fit(final_dt_lasso4lch, data = df_train_dt)
```

``` r
# Apply model in the testing data
results_dt_lasso4lch <- 
  predict(fit_dt_lasso4lch, df_test_dt, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_dt, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_dt_lasso4lch <- conf_mat(results_dt_lasso4lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_dt_lasso4lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.908
#>  2 kap                  binary         0.668
#>  3 sens                 binary         0.701
#>  4 spec                 binary         0.951
#>  5 ppv                  binary         0.746
#>  6 npv                  binary         0.939
#>  7 mcc                  binary         0.668
#>  8 j_index              binary         0.652
#>  9 bal_accuracy         binary         0.826
#> 10 detection_prevalence binary         0.160
#> 11 precision            binary         0.746
#> 12 recall               binary         0.701
#> 13 f_meas               binary         0.723
```

``` r
# AUC value
roc_auc(results_dt_lasso4lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.888
```

``` r
# Plot ROC
roc_curve(results_dt_lasso4lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>% 
  ggplot(aes(x = 1 - specificity,
             y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-192-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_dt_lasso4lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-193-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
last_fit(final_dt_lasso4lch, split = df_split_dt) %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 10)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-194-1.png" width="100%" />

#### Support Vector Machine

``` r
# Prepare dataset for SVM
df_mlb_svm <- df_mlb %>% 
  select(-name, -highLevel, -school) %>% # Remove variables of name, high-level and school
  fastDummies::dummy_cols(select_columns = c( 'team', 'schooltype', 'bats', 'throws', 'position', 'sch_reg', 'birth_place'), remove_selected_columns = TRUE) # Converting the categorical variable into dummies

str(df_mlb_svm)
#> tibble [4,321 × 222] (S3: tbl_df/tbl/data.frame)
#>  $ height              : num [1:4321] 191 191 183 175 183 188 185 183 191 191 ...
#>  $ weight              : num [1:4321] 95 99 82 86 86 104 86 98 86 88 ...
#>  $ birth_year          : num [1:4321] 1992 1992 1989 1992 1989 ...
#>  $ age                 : num [1:4321] 18 18 21 18 21 22 18 19 18 21 ...
#>  $ year                : num [1:4321] 2010 2010 2010 2010 2010 2010 2010 2010 2010 2010 ...
#>  $ round               : num [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ overall_pick        : num [1:4321] 1 3 4 8 10 12 15 17 18 20 ...
#>  $ mlb_debut           : Factor w/ 2 levels "no","yes": 2 2 2 2 2 2 1 1 2 1 ...
#>  $ g                   : num [1:4321] 139 222 944 666 788 198 660 224 931 304 ...
#>  $ ab                  : num [1:4321] 486 832 3503 2582 2952 ...
#>  $ r                   : num [1:4321] 83 116 489 483 478 128 301 93 508 148 ...
#>  $ h                   : num [1:4321] 146 224 990 683 822 214 494 185 949 292 ...
#>  $ b2                  : num [1:4321] 30 51 167 115 180 55 92 35 203 62 ...
#>  $ b3                  : num [1:4321] 4 12 12 29 9 0 14 11 35 12 ...
#>  $ hr                  : num [1:4321] 23 23 71 49 126 22 35 18 80 8 ...
#>  $ rbi                 : num [1:4321] 75 116 438 280 513 112 242 105 536 103 ...
#>  $ sb                  : num [1:4321] 29 25 119 294 33 0 73 11 125 34 ...
#>  $ cs                  : num [1:4321] 10 10 56 75 13 2 33 13 40 9 ...
#>  $ bb                  : num [1:4321] 76 97 348 374 325 113 307 109 351 104 ...
#>  $ so                  : num [1:4321] 106 147 400 647 827 165 645 212 821 288 ...
#>  $ sh                  : num [1:4321] 2 1 75 52 0 0 13 0 9 0 ...
#>  $ sf                  : num [1:4321] 2 4 38 20 26 9 17 8 38 11 ...
#>  $ hbp                 : num [1:4321] 3 7 54 44 59 8 17 4 22 15 ...
#>  $ gdp                 : num [1:4321] 10 31 115 28 48 14 29 13 74 29 ...
#>  $ tb                  : num [1:4321] 253 368 1394 1003 1398 ...
#>  $ pa                  : num [1:4321] 569 941 4018 3072 3362 ...
#>  $ xbh                 : num [1:4321] 57 86 250 193 315 77 141 64 318 82 ...
#>  $ avg                 : num [1:4321] 0.3 0.269 0.283 0.265 0.278 0.31 0.226 0.238 0.268 0.258 ...
#>  $ obp                 : num [1:4321] 0.397 0.349 0.353 0.365 0.359 0.408 0.324 0.332 0.334 0.326 ...
#>  $ slg                 : num [1:4321] 0.521 0.442 0.398 0.388 0.474 0.485 0.329 0.381 0.412 0.356 ...
#>  $ ops                 : num [1:4321] 0.917 0.791 0.751 0.753 0.832 0.893 0.653 0.714 0.746 0.682 ...
#>  $ babip               : num [1:4321] 0.343 0.302 0.299 0.333 0.344 0.374 0.302 0.301 0.324 0.336 ...
#>  $ bmi                 : num [1:4321] 26 27.1 24.5 28.1 25.7 ...
#>  $ hr_ab               : num [1:4321] 0.0473 0.0276 0.0203 0.019 0.0427 ...
#>  $ iso                 : num [1:4321] 0.221 0.173 0.115 0.123 0.196 0.175 0.103 0.143 0.144 0.098 ...
#>  $ bb_so               : num [1:4321] 0.717 0.66 0.87 0.578 0.393 ...
#>  $ sbr                 : num [1:4321] 0.744 0.714 0.68 0.797 0.717 ...
#>  $ team_ANA            : int [1:4321] 0 0 0 0 0 0 0 0 1 0 ...
#>  $ team_ARI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_ATL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_BAL            : int [1:4321] 0 1 0 0 0 0 0 0 0 0 ...
#>  $ team_BOS            : int [1:4321] 0 0 0 0 0 0 0 0 0 1 ...
#>  $ team_CHA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CHN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_CIN            : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ team_CLE            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_COL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_DET            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_HOU            : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ team_KCA            : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ team_LAN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIL            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MIN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_MON            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_NYN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_OAK            : int [1:4321] 0 0 0 0 1 0 0 0 0 0 ...
#>  $ team_PHI            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_PIT            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SDN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SEA            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SFN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_SLN            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_TBA            : int [1:4321] 0 0 0 0 0 0 0 1 0 0 ...
#>  $ team_TEX            : int [1:4321] 0 0 0 0 0 0 1 0 0 0 ...
#>  $ team_TOR            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ team_WAS            : int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ schooltype_4Year    : int [1:4321] 0 0 1 0 1 1 0 0 0 1 ...
#>  $ schooltype_HS       : int [1:4321] 0 1 0 1 0 0 1 1 1 0 ...
#>  $ schooltype_JrCollege: int [1:4321] 1 0 0 0 0 0 0 0 0 0 ...
#>  $ bats_B              : int [1:4321] 0 0 0 0 0 1 0 0 1 0 ...
#>  $ bats_L              : int [1:4321] 1 0 0 0 0 0 1 1 0 0 ...
#>  $ bats_R              : int [1:4321] 0 1 1 1 1 0 0 0 0 1 ...
#>  $ throws_L            : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ throws_R            : int [1:4321] 1 1 1 1 1 1 1 1 1 1 ...
#>  $ position_1B         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_2B         : int [1:4321] 0 0 0 1 0 0 0 0 0 0 ...
#>  $ position_3B         : int [1:4321] 0 1 0 0 0 0 0 0 1 0 ...
#>  $ position_C          : int [1:4321] 0 0 0 0 0 1 0 0 0 0 ...
#>  $ position_CF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_IF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_LF         : int [1:4321] 1 0 0 0 0 0 0 1 0 0 ...
#>  $ position_OF         : int [1:4321] 0 0 0 0 1 0 1 0 0 1 ...
#>  $ position_RF         : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ position_SS         : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AB          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AL          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AR          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_AZ          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_BC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CA          : int [1:4321] 0 0 1 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CO          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CT          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_CU          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DC          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_DE          : int [1:4321] 0 0 0 0 0 0 0 0 0 0 ...
#>  $ sch_reg_FL          : int [1:4321] 0 1 0 0 0 1 0 0 0 0 ...
#>  $ sch_reg_GA          : int [1:4321] 0 0 0 1 0 0 1 0 1 0 ...
#>   [list output truncated]
```

``` r
# Set the seed
set.seed(123)

# Split the data into train and test
df_split_svm <- initial_split(df_mlb_svm, strata = mlb_debut) #strata argument can help the training and test data sets will keep roughly the same proportions of mlb_debut = yes and no as in the original data.
df_train_svm <- training(df_split_svm)
df_test_svm <- testing(df_split_svm)
```

``` r
# create cross-validation resample for tuning the model.
set.seed(123)
dfa_folds_svm <- vfold_cv(df_train_svm, v = 5, strata = mlb_debut)
```

``` r
# Create a SVM model object
svm <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune(),
  margin = NULL,
) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification") %>% 
  translate()
```

``` r
# Set up possible values for these hyperparameters to try
svm_grid <-grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5)

svm_grid
#> # A tibble: 25 × 2
#>         cost    rbf_sigma
#>        <dbl>        <dbl>
#>  1  0.000977 0.0000000001
#>  2  0.0131   0.0000000001
#>  3  0.177    0.0000000001
#>  4  2.38     0.0000000001
#>  5 32        0.0000000001
#>  6  0.000977 0.0000000316
#>  7  0.0131   0.0000000316
#>  8  0.177    0.0000000316
#>  9  2.38     0.0000000316
#> 10 32        0.0000000316
#> # … with 15 more rows
```

#### SVM model with all variables

``` r
# Create recipe
recipe_svm_all <- 
  recipe(mlb_debut ~ ., data = df_train_svm) %>% 
  step_zv(all_predictors())
```

``` r
# Create workflow
svm_workflow_all <-
  workflow() %>% 
  add_recipe(recipe_svm_all) %>%
  add_model(svm)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
svm_reg_all <- tune_grid(
  svm_workflow_all,
  resamples = dfa_folds_svm,
  grid = svm_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_svm_all <- svm_reg_all %>%
  select_best("roc_auc")

best_svm_all
#> # A tibble: 1 × 3
#>    cost rbf_sigma .config              
#>   <dbl>     <dbl> <chr>                
#> 1  2.38   0.00316 Preprocessor1_Model19
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_wf_svm_all <- 
  svm_workflow_all %>% 
  finalize_workflow(best_svm_all)
```

``` r
# Fit the tuned model in training data
final_fit_svm_all <- 
  fit(final_wf_svm_all, data = df_train_svm) 
```

``` r
# Apply model in the testing data
results_svm_prob_all <- 
  predict(final_fit_svm_all, df_test_svm, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_svm, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_svm_all <- conf_mat(results_svm_prob_all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_svm_all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.910
#>  2 kap                  binary         0.658
#>  3 sens                 binary         0.647
#>  4 spec                 binary         0.964
#>  5 ppv                  binary         0.788
#>  6 npv                  binary         0.930
#>  7 mcc                  binary         0.662
#>  8 j_index              binary         0.611
#>  9 bal_accuracy         binary         0.806
#> 10 detection_prevalence binary         0.140
#> 11 precision            binary         0.788
#> 12 recall               binary         0.647
#> 13 f_meas               binary         0.710
```

``` r
# AUC
roc_auc(results_svm_prob_all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.955
```

``` r
# Plot ROC
roc_curve(results_svm_prob_all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>%
    ggplot(aes(x = 1 - specificity,
               y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() +
      theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-209-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_svm_prob_all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-210-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
set.seed(123)
last_fit(final_wf_svm_all, split = df_split_svm) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(method = "permute", 
      target = 'mlb_debut', 
      metric = "accuracy",
      num_features = 10,
      pred_wrapper = kernlab::predict, train = df_train_svm)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-211-1.png" width="100%" />

#### SVM model with variables of lasso4all

``` r
# Create recipe
recipe_svm_lasso4all <- 
  recipe(mlb_debut ~ ., data = df_train_svm) %>% 
  step_rm(weight, year, round, g, r, b2, bb, pa, obp, slg, hr_ab, iso) %>% 
  step_zv(all_predictors())
```

``` r
# Create workflow
svm_workflow_lasso4all <-
  workflow() %>% 
  add_recipe(recipe_svm_lasso4all) %>%
  add_model(svm)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
svm_reg_lasso4all <- tune_grid(
  svm_workflow_lasso4all,
  resamples = dfa_folds_svm,
  grid = svm_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_svm_lasso4all <- svm_reg_lasso4all %>%
  select_best("roc_auc")

best_svm_lasso4all
#> # A tibble: 1 × 3
#>    cost rbf_sigma .config              
#>   <dbl>     <dbl> <chr>                
#> 1  2.38   0.00316 Preprocessor1_Model19
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_wf_svm_lasso4all <- 
  svm_workflow_lasso4all %>% 
  finalize_workflow(best_svm_lasso4all)
```

``` r
# Fit the tuned model in training data
final_fit_svm_lasso4all <- 
  fit(final_wf_svm_lasso4all, data = df_train_svm) 
```

``` r
# Apply model in the testing data
results_svm_prob_lasso4all <- 
  predict(final_fit_svm_lasso4all, df_test_svm, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_svm, Predicted_Probability = .) %>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_svm_lasso4all <- conf_mat(results_svm_prob_lasso4all, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_svm_lasso4all, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.903
#>  2 kap                  binary         0.626
#>  3 sens                 binary         0.614
#>  4 spec                 binary         0.962
#>  5 ppv                  binary         0.769
#>  6 npv                  binary         0.924
#>  7 mcc                  binary         0.632
#>  8 j_index              binary         0.576
#>  9 bal_accuracy         binary         0.788
#> 10 detection_prevalence binary         0.136
#> 11 precision            binary         0.769
#> 12 recall               binary         0.614
#> 13 f_meas               binary         0.683
```

``` r
# AUC value
roc_auc(results_svm_prob_lasso4all, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.952
```

``` r
# Plot ROC
roc_curve(results_svm_prob_lasso4all, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>%
    ggplot(aes(x = 1 - specificity,
               y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() +
      theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-221-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_svm_prob_lasso4all %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-222-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
set.seed(123)
last_fit(final_wf_svm_lasso4all, split = df_split_svm) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(method = "permute", 
      target = 'mlb_debut', 
      metric = "accuracy",
      num_features = 10,
      pred_wrapper = kernlab::predict, train = df_train_svm)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-223-1.png" width="100%" />

#### SVM model with expert-selected variables

``` r
# Create recipe
recipe_svm_selected_lch<- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi  + round + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + obp + slg + ops + iso + bb_so  + hr_ab +sbr + ab, data = df_train_svm) %>% 
  step_zv(all_predictors())
```

``` r
# Create workflow
svm_workflow_selected_lch <-
  workflow() %>% 
  add_recipe(recipe_svm_selected_lch) %>%
  add_model(svm)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
svm_reg_selected_lch <- tune_grid(
  svm_workflow_selected_lch,
  resamples = dfa_folds_svm,
  grid = svm_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_svm_selected_lch <- svm_reg_selected_lch %>%
  select_best("roc_auc")

best_svm_selected_lch
#> # A tibble: 1 × 3
#>    cost rbf_sigma .config              
#>   <dbl>     <dbl> <chr>                
#> 1    32   0.00316 Preprocessor1_Model20
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_wf_svm_selected_lch <- 
  svm_workflow_selected_lch %>% 
  finalize_workflow(best_svm_selected_lch)
```

``` r
# Fit the tuned model in training data
final_fit_svm_selected_lch <- 
  fit(final_wf_svm_selected_lch, data = df_train_svm) 
```

``` r
# Apply model in the testing data
results_svm_prob_selected_lch <- 
  predict(final_fit_svm_selected_lch, df_test_svm, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_svm, Predicted_Probability = .)%>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_svm_selected_lch <- conf_mat(results_svm_prob_selected_lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_svm_selected_lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.913
#>  2 kap                  binary         0.668
#>  3 sens                 binary         0.652
#>  4 spec                 binary         0.967
#>  5 ppv                  binary         0.8  
#>  6 npv                  binary         0.931
#>  7 mcc                  binary         0.673
#>  8 j_index              binary         0.619
#>  9 bal_accuracy         binary         0.809
#> 10 detection_prevalence binary         0.139
#> 11 precision            binary         0.8  
#> 12 recall               binary         0.652
#> 13 f_meas               binary         0.719
```

``` r
# AUC value
roc_auc(results_svm_prob_selected_lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.956
```

``` r
# Plot ROC
roc_curve(results_svm_prob_selected_lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>%
    ggplot(aes(x = 1 - specificity,
               y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() +
      theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-233-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_svm_prob_selected_lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-234-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
set.seed(123)
last_fit(final_wf_svm_selected_lch, split = df_split_svm) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(method = "permute", 
      target = 'mlb_debut', 
      metric = "accuracy",
      num_features = 10,
      pred_wrapper = kernlab::predict, train = df_train_svm)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-235-1.png" width="100%" />

#### SVM model with variables of Lasso4lch

``` r
# Create recipe
recipe_svm_lasso4lch<- 
  recipe(mlb_debut ~ bats_B + bats_L + bats_R + age + bmi + overall_pick + team_ANA + team_ARI + team_ATL + team_BAL + team_BOS + team_CHA + team_CHN + team_CIN + team_CLE + team_COL + team_DET + team_HOU + team_KCA + team_LAN + team_MIA + team_MIL +  team_MIN + team_MON + team_NYA + team_NYN + team_OAK + team_PHI + team_PIT + team_SDN + team_SEA + team_SFN + team_SLN + team_TBA + team_TEX + team_TOR + team_WAS + avg + slg + ops + bb_so +sbr + ab, data = df_train_svm) %>% 
  step_zv(all_predictors())
```

``` r
# Create workflow
svm_workflow_lasso4lch <-
  workflow() %>% 
  add_recipe(recipe_svm_lasso4lch) %>%
  add_model(svm)
```

``` r
doParallel::registerDoParallel(cores = parallel::detectCores()) #to parallel process and speed up tuning

set.seed(123)
svm_reg_lasso4lch <- tune_grid(
  svm_workflow_lasso4lch,
  resamples = dfa_folds_svm,
  grid = svm_grid,
  control = control_grid(save_pred = TRUE))

doParallel::stopImplicitCluster() # to stop parallel processing
```

``` r
# Selecting the best hyperparameter by the value of AUC
best_svm_lasso4lch <- svm_reg_lasso4lch %>%
  select_best("roc_auc")

best_svm_lasso4lch
#> # A tibble: 1 × 3
#>    cost rbf_sigma .config              
#>   <dbl>     <dbl> <chr>                
#> 1    32   0.00316 Preprocessor1_Model20
```

``` r
# Finalize the tuneable workflow with the best parameter values.
final_wf_svm_lasso4lch <- 
  svm_workflow_lasso4lch %>% 
  finalize_workflow(best_svm_lasso4lch)
```

``` r
# Fit the tuned model in training data
final_fit_svm_lasso4lch <- 
  fit(final_wf_svm_lasso4lch, data = df_train_svm) 
```

``` r
# Apply model in the testing data
results_svm_prob_lasso4lch <- 
  predict(final_fit_svm_lasso4lch, df_test_svm, type = 'prob') %>% 
  pluck(2) %>% 
  bind_cols(df_test_svm, Predicted_Probability = .)%>% 
  mutate(predictedClass = as.factor(ifelse(Predicted_Probability > 0.5, 'yes', 'no')))
```

``` r
# Print out performances of confusion matrix
conf_mat_svm_lasso4lch <- conf_mat(results_svm_prob_lasso4lch, truth = mlb_debut, estimate = predictedClass)

summary(conf_mat_svm_lasso4lch, event_level='second')
#> # A tibble: 13 × 3
#>    .metric              .estimator .estimate
#>    <chr>                <chr>          <dbl>
#>  1 accuracy             binary         0.912
#>  2 kap                  binary         0.667
#>  3 sens                 binary         0.658
#>  4 spec                 binary         0.964
#>  5 ppv                  binary         0.791
#>  6 npv                  binary         0.932
#>  7 mcc                  binary         0.671
#>  8 j_index              binary         0.622
#>  9 bal_accuracy         binary         0.811
#> 10 detection_prevalence binary         0.142
#> 11 precision            binary         0.791
#> 12 recall               binary         0.658
#> 13 f_meas               binary         0.718
```

``` r
# AUC value
roc_auc(results_svm_prob_lasso4lch, truth = mlb_debut, Predicted_Probability, event_level = 'second')
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.956
```

``` r
# Plot ROC
roc_curve(results_svm_prob_lasso4lch, truth = mlb_debut,
          Predicted_Probability,
          event_level = 'second') %>%
    ggplot(aes(x = 1 - specificity,
               y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() +
      theme_bw()
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-245-1.png" width="100%" />

``` r
# Plot predicted probability vs OPS
results_svm_prob_lasso4lch %>% 
ggplot()+
  geom_point(aes(x = ops, y = Predicted_Probability, color=mlb_debut))+
  labs(x = "OPS", y = "Predicted Probability", color = "MLB")+
  theme_light()+
  scale_color_manual(values = c("#FFD520", "#E03A3E"), limits = c("yes", "no"))
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-246-1.png" width="100%" />

``` r
# Find out top10 most important variables in this model
set.seed(123)
last_fit(final_wf_svm_lasso4lch, split = df_split_svm) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip::vip(method = "permute", 
      target = 'mlb_debut', 
      metric = "accuracy",
      num_features = 10,
      pred_wrapper = kernlab::predict, train = df_train_svm)
```

<img src="fig_From MiLB to MLB Debut - a Machine Learning Application/figures/README-unnamed-chunk-247-1.png" width="100%" />
