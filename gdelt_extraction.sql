WITH candidate_articles AS (
    SELECT
        -- Unique GDELT record ID
        GKGRECORDID,

        -- GDELT record-creation time (UTC), not publication time
        SAFE.PARSE_TIMESTAMP(
            '%Y%m%d%H%M%S',
            SUBSTR(GKGRECORDID, 1, 14),
            'UTC'
        ) AS capture_timestamp_utc,

        -- Restrict source to actual web articles
        SourceCollectionIdentifier,

        -- Preserve the original GDELT URL
        DocumentIdentifier AS article_url_raw,

        -- Preserve organization evidence
        V2Organizations AS raw_organizations,

        -- Preserve the complete original tone field (Nulls included)
        V2Tone AS raw_v2tone,

        -- Extract first numerical tone component safely
        SAFE_CAST(
            SPLIT(V2Tone, ',')[SAFE_OFFSET(0)]
            AS FLOAT64
        ) AS first_tone
    FROM
        `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        -- 1. Partition pruning (Dec 31, 2024 to Sep 30, 2026)
        _PARTITIONTIME >= TIMESTAMP('2024-12-31 00:00:00+00')
        AND _PARTITIONTIME < TIMESTAMP('2026-10-01 00:00:00+00')

        -- 2. Only actual web documents
        AND SourceCollectionIdentifier = 1

        -- 3. Valid source URLs
        AND NULLIF(TRIM(DocumentIdentifier), '') IS NOT NULL

        -- 4. Valid GDELT capture-time identifier
        AND REGEXP_CONTAINS(GKGRECORDID, r'^\d{14}-')

        -- 5. Exact organization matching (Corrected for GDELT comma offsets)
        AND EXISTS (
            SELECT 1
            FROM UNNEST(
                SPLIT(COALESCE(V2Organizations, ''), ';')
            ) AS org_with_offset
            -- Split by the comma and take the first element to isolate the name
            WHERE LOWER(TRIM(SPLIT(org_with_offset, ',')[SAFE_OFFSET(0)])) IN (
                'reliance industries',
                'reliance industries limited'
            )
        )
),
deduplicated_articles AS (
    SELECT *
    FROM candidate_articles
    WHERE capture_timestamp_utc IS NOT NULL

    -- Retain the earliest matching GDELT observation
    -- for each exact source URL within the extraction period.
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY article_url_raw
        ORDER BY
            capture_timestamp_utc ASC,
            GKGRECORDID ASC
    ) = 1
)

SELECT
    GKGRECORDID,
    capture_timestamp_utc,
    SourceCollectionIdentifier,
    article_url_raw,
    raw_organizations,
    raw_v2tone,
    first_tone
FROM deduplicated_articles
ORDER BY
    capture_timestamp_utc ASC,
    article_url_raw ASC;