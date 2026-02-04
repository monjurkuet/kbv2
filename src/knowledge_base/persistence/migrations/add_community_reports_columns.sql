-- Add community reports columns to documents, entities, and edges tables
ALTER TABLE documents ADD COLUMN IF NOT EXISTS community_reports_generated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS total_communities INTEGER NOT NULL DEFAULT 0;

ALTER TABLE entities ADD COLUMN IF NOT EXISTS community_reports_generated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS total_communities INTEGER NOT NULL DEFAULT 0;

ALTER TABLE edges ADD COLUMN IF NOT EXISTS community_reports_generated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE edges ADD COLUMN IF NOT EXISTS total_communities INTEGER NOT NULL DEFAULT 0;
