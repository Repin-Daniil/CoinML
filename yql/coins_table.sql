CREATE TABLE coins (
    coin_id String,

    source_url String,
    image_url_obverse String,
    image_url_reverse String,

    status String,         -- 'new_url', 'scraped_metadata', 'selected_for_download', 'processed', 'failed'

    s3_path_obverse String,
    s3_path_reverse String,

    -- For DataLens
    title String,
    condition Int8,
    metal String,
    country String,
    nominal String,
    coin_year Int32,

    -- Utility
    retry_count Uint8,
    scraped_at Timestamp,
    processed_at Timestamp,
    INDEX status_index GLOBAL SYNC ON (status),
    PRIMARY KEY (coin_id)
)