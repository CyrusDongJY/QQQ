-- QQQ live research signal metadata.
-- Safe to rerun in Supabase SQL Editor.

BEGIN;

-- These fields are deliberately separate from call_wall/put_wall. Existing
-- walls are near-expiry gamma-weighted levels; the research signal requires
-- max open interest by right within one standard monthly expiration.
ALTER TABLE stock_options_pre_market
    ADD COLUMN IF NOT EXISTS monthly_wall_expiry date,
    ADD COLUMN IF NOT EXISTS monthly_call_wall numeric,
    ADD COLUMN IF NOT EXISTS monthly_put_wall numeric,
    ADD COLUMN IF NOT EXISTS monthly_call_wall_oi bigint,
    ADD COLUMN IF NOT EXISTS monthly_put_wall_oi bigint,
    ADD COLUMN IF NOT EXISTS monthly_wall_oi_source_date date,
    ADD COLUMN IF NOT EXISTS monthly_wall_method text,
    ADD COLUMN IF NOT EXISTS monthly_wall_quality text;

CREATE INDEX IF NOT EXISTS idx_stock_options_pre_market_monthly_wall_source
    ON stock_options_pre_market (ticker, monthly_wall_oi_source_date DESC);

COMMENT ON COLUMN stock_options_pre_market.monthly_wall_expiry IS
    'Next standard monthly option settlement used for QQQ wall research.';
COMMENT ON COLUMN stock_options_pre_market.monthly_call_wall IS
    'Call strike with maximum aggregate OI inside monthly_wall_expiry.';
COMMENT ON COLUMN stock_options_pre_market.monthly_put_wall IS
    'Put strike with maximum aggregate OI inside monthly_wall_expiry.';
COMMENT ON COLUMN stock_options_pre_market.monthly_wall_oi_source_date IS
    'Market date represented by the overnight OI snapshot; must equal signal_date.';
COMMENT ON COLUMN stock_options_pre_market.monthly_wall_method IS
    'Versioned wall construction method, currently MAX_OI_BY_RIGHT_STANDARD_MONTHLY_V1.';

-- Nullable until the full live five-factor pipeline is deployed. Unknown or
-- missing values remain WATCH in the Builder; no default is assigned.
ALTER TABLE stock_spot_post_close
    ADD COLUMN IF NOT EXISTS five_factor_regime text,
    ADD COLUMN IF NOT EXISTS five_factor_regime_reason text,
    ADD COLUMN IF NOT EXISTS five_factor_regime_source text,
    ADD COLUMN IF NOT EXISTS five_factor_regime_as_of timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'stock_spot_post_close_five_factor_regime_check'
    ) THEN
        ALTER TABLE stock_spot_post_close
            ADD CONSTRAINT stock_spot_post_close_five_factor_regime_check
            CHECK (
                five_factor_regime IS NULL OR five_factor_regime IN (
                    'high_rising',
                    'high_falling',
                    'vol_rising',
                    'low_calm',
                    'normalized',
                    'transition_no_trade'
                )
            );
    END IF;
END $$;

COMMIT;
