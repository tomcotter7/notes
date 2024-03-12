# Golang (Go)

## Creating a Go Module
`go mod init <module-name>`.
If you want to use that folder (module) in a different folder you can do `go mod edit -replace <module-name>=<path-to-module>`. Finally, run `go mod tidy` to clean up any imports.

