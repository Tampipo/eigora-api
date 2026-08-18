# Changelog

## [0.2.1](https://github.com/Tampipo/eigora-api/compare/v0.2.0...v0.2.1) (2026-08-18)


### Bug Fixes

* derive app version from package metadata instead of a hardcoded … ([517b0dc](https://github.com/Tampipo/eigora-api/commit/517b0dc6859e823ea5f77989c905c1e27e1f3fff))
* derive app version from package metadata instead of a hardcoded string ([f408b74](https://github.com/Tampipo/eigora-api/commit/f408b74044b8e596f8f9eecacb4e1195a13fde55))

## [0.2.0](https://github.com/Tampipo/eigora-api/compare/v0.1.0...v0.2.0) (2026-08-18)


### ⚠ BREAKING CHANGES

* all endpoints move from /qm/* to /v1/qm/*, including the /qm/evolve WebSocket. Existing callers (eigora-web) must update their base path accordingly.

### Features

* version API routes under /v1 ([ecd03b9](https://github.com/Tampipo/eigora-api/commit/ecd03b908381dafd6df1987202c641f023f604d2))

## [0.2.0](https://github.com/Tampipo/eigora-api/compare/eigora-api-v0.1.0...eigora-api-v0.2.0) (2026-08-18)


### ⚠ BREAKING CHANGES

* all endpoints move from /qm/* to /v1/qm/*, including the /qm/evolve WebSocket. Existing callers (eigora-web) must update their base path accordingly.

### Features

* version API routes under /v1 ([ecd03b9](https://github.com/Tampipo/eigora-api/commit/ecd03b908381dafd6df1987202c641f023f604d2))
