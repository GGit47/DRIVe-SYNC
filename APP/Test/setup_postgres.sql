-- DRIVe-SYNC PostgreSQL Setup
-- Run this once: psql -U postgres -f setup_postgres.sql

CREATE DATABASE drivesync;
\c drivesync

CREATE TABLE IF NOT EXISTS obd_readings (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ DEFAULT NOW(),
    scenario     TEXT,
    rpm          REAL,
    speed_kmh    REAL,
    throttle_pct REAL,
    coolant_temp REAL,
    fuel_level   REAL,
    maf_gs       REAL,
    o2_voltage   REAL,
    dtc_codes    TEXT,
    style_label  TEXT,
    style_conf   REAL
);

CREATE TABLE IF NOT EXISTS predictions (
    id           SERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ DEFAULT NOW(),
    model_used   TEXT,
    speed        REAL,
    acceleration REAL,
    risk_score   INTEGER,
    harsh_brake  INTEGER,
    harsh_accel  INTEGER,
    sharp_turn   INTEGER,
    is_night     INTEGER,
    is_speeding  INTEGER,
    label        TEXT,
    confidence   REAL,
    prob_unsafe  REAL
);

-- Helpful views
CREATE VIEW unsafe_sessions AS
  SELECT ts, speed, risk_score, confidence, model_used
  FROM predictions
  WHERE label = 'Unsafe'
  ORDER BY ts DESC;

CREATE VIEW obd_summary AS
  SELECT
    scenario,
    COUNT(*)               AS readings,
    AVG(rpm)               AS avg_rpm,
    AVG(speed_kmh)         AS avg_speed,
    AVG(coolant_temp)      AS avg_coolant,
    COUNT(*) FILTER (WHERE style_label='Aggressive') AS aggressive_count
  FROM obd_readings
  GROUP BY scenario;

\echo 'DRIVe-SYNC database ready.'
