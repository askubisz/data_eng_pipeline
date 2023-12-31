# Changelog

## 2.0.1 (2022-08-28)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v2.0.0...v2.0.1)

Update to Grafana 9.

## 2.0.0 (2021-11-19)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v1.0.0...v2.0.0)

### Enhancements

- Add option for 120 minute intervals ([#40](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/pull/40)) (thanks [@ChrizZz90](https://github.com/ChrizZz90)!)
- BREAKING CHANGE: Update to Grafana 8 theme API

## 1.0.0 (2021-06-15)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v0.10.0...v1.0.0)

### Enhancements

- Set color for null values ([#24](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/pull/24)) (thanks [@KatrinaTurner](https://github.com/KatrinaTurner)!)
- Fix dates in changelog ([#26](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/pull/26)) (thanks [@dnrce](https://github.com/dnrce)!)
- Fix display processor bug in Grafana 8

## 0.10.0 (2021-02-16)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v0.9.1...v0.10.0)

### Enhancements

- Make dimensions clearable
- Add fallback panel for unsupported Grafana versions
- Add wizard for configuring the query

## 0.9.1 (2021-01-13)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v0.9.0...v0.9.1)

### Bug fixes

- Min and max were incorrectly calculated for aggregations ([#16](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/issues/16))

## 0.9.0 (2020-12-07)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v0.8.1...v0.9.0)

### Enhancements

- Highlight the legend section for the selected hour ([#15](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/issues/15))
- Select the fields to use for time and value
- Add option to configure legend quality
- Add option to use the new color scheme field option in Grafana 7.3+
- Add option to show a cell border ([#17](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/issues/17))
- Add option to disable tooltips

## 0.8.1 (2020-11-27)

[Full changelog](https://github.com/marcusolsson/grafana-hourly-heatmap-panel/compare/v0.8.0...v0.8.1)

### Enhancements

- Updated `@grafana` dependencies from `^7.0.0` to `^7.3.0`
- Improved release process using the new [GitHub workflows](https://github.com/grafana/plugin-workflows) for Grafana plugins
