# Kasa recovery selector map (credentials intentionally omitted)

Use the configured power-control client and verify the target node identity after recovery.
Do not place passwords, account names, private addresses, or cloud tokens in this package.

| Hardware role | Selector | Plug |
|---|---:|---:|
| compute node 1 | 9067 | 1 |
| compute node 2 | 9067 | 2 |
| compute node 3 | 9067 | 3 |
| compute node 4 | 9067 | 4 |
| compute node work | 9067 | 5 |
| compute node 6 | 9067 | 6 |
| fabric switch | strip-1 | 1 |
| compute node 7 | strip-1 | 4 |
| compute node 8 | strip-1 | 5 |

Recovery pattern: `<power-client> cycle <selector> <plug> 30`, then wait for boot and
confirm the expected hostname from an independent connection. Never cycle the control host,
storage appliance, or fabric switch casually.
