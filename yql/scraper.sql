-- Get Batch
SELECT coin_id, source_url, retry_count
FROM coins
WHERE status = "new"
LIMIT 50;

-- Update metadata
UPDATE coins
SET
    status = "scrapped_metadata",
    metal = $metal,
    nominal = $nominal,
    coin_year = $coin_year,
    status = "scrapped_metadata"
WHERE coin_id = $coin_id;

-- Error case
UPDATE coins
SET
    retry_count = retry_count + 1,
    status = CASE
        WHEN retry_count + 1 >= 5 THEN "failed"
        ELSE status
    END
WHERE coin_id = $coin_id;
