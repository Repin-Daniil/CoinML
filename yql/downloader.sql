-- Get Batch
SELECT coin_id, image_url_obverse, image_url_reverse, retry_count
FROM coins
WHERE status = 'selected_for_download'
LIMIT 50

-- Save s3 paths
UPDATE coins
SET
    s3_path_obverse=$obverse,
    s3_path_reverse=$reverse,
    status = "Saved"
WHERE coin_id = $coin_id

-- Error case
UPDATE coins
SET
    retry_count = retry_count + 1,
    status = CASE
        WHEN retry_count + 1 >= 5 THEN "failed"
        ELSE status
    END
WHERE coin_id = $coin_id;
