# P958 exact-assignment recovery/publication status

Status: **PENDING — hashes available; assignment payload not yet public**

The definitive P931 solver summary preserves these accepted integrity pins:

- assignment document SHA-256: `b8f07185e54f018af4bcc2b6831b457b1ada2c97ad166226ee19e3e8e91bbd8d`
- canonical assignment-map SHA-256: `c260b1a05ad1368a9aa11ee184fbe0a2734e781c7bbf4e1bf40c67821ea8786c`
- independent verification receipt SHA-256: `60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a`
- reviewed 13-entry source-artifact manifest SHA-256: `d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a`

The 22,016-cell assignment payload was not durably copied into the public package before its source scratch was reclaimed. The pins above therefore prove identity only if matching bytes are later recovered or deterministically regenerated; they do not make identity-by-identity replay possible from this release.

## Publication gate

P958 may change this status to complete only after all of the following are public and independently checked:

1. the exact 22,016-cell assignment document and canonical map;
2. SHA-256 equality to both accepted pins above, or a clearly labeled replacement solve with new lineage;
3. an executable verifier that closes legal menu, cell identities, exact bytes/slack, objective, six-class predictions, full tier counts, P922 join, and P928 single application;
4. a complete output manifest whose bytes rehash to the published manifest pin;
5. package, privacy, source-manifest, and unauthenticated readback checks.

Until those gates pass, `artifacts/P931_V3_DEFINITIVE.public.json` remains a projected, reviewed summary with `assignment_payload_redistributed=false`; it is not a downloadable assignment or a direct TRUE-C measurement.
