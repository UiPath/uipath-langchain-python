-- S2P Procurement Ontology — physical tables + sample data.
-- Column names are quoted to preserve exact case that the YARRRML mapping binds to.
-- The runtime strips the `datafabric.` source alias; Ontop emits double-quoted,
-- case-exact SQL, so physical columns must match.

-- ============================================================
-- Drop order respects foreign key dependencies
-- ============================================================

DROP TABLE IF EXISTS "SpendRecord";
DROP TABLE IF EXISTS "ExceptionRule";
DROP TABLE IF EXISTS "ToleranceException";
DROP TABLE IF EXISTS "Budget";
DROP TABLE IF EXISTS "Item";
DROP TABLE IF EXISTS "ContractScope";
DROP TABLE IF EXISTS "Contract";
DROP TABLE IF EXISTS "Invoice";
DROP TABLE IF EXISTS "PurchaseOrder";
DROP TABLE IF EXISTS "Requisition";
DROP TABLE IF EXISTS "CostCenter";
DROP TABLE IF EXISTS "FXRate";
DROP TABLE IF EXISTS "Commodity";
DROP TABLE IF EXISTS "Supplier";

-- ============================================================
-- Supplier — canonical identity with hierarchy
-- ============================================================

CREATE TABLE "Supplier" (
  "supplierId"        VARCHAR(20)  PRIMARY KEY,
  "name"              VARCHAR(200) NOT NULL,
  "taxId"             VARCHAR(40),
  "status"            VARCHAR(20)  NOT NULL,   -- active | inactive | blocked
  "parentSupplierId"  VARCHAR(20),             -- self-ref for hierarchy
  "country"           VARCHAR(60)  NOT NULL,
  "preferredFlag"     BOOLEAN      NOT NULL DEFAULT FALSE
);

INSERT INTO "Supplier" VALUES
  ('SUP-001', 'Acme Global Corp',        'US-EIN-12-3456789', 'active',   NULL,      'US',  TRUE),
  ('SUP-002', 'Acme EMEA GmbH',          'DE-VAT-987654321',  'active',   'SUP-001', 'DE',  TRUE),
  ('SUP-003', 'Acme APAC Pte Ltd',       'SG-UEN-202012345',  'active',   'SUP-001', 'SG',  TRUE),
  ('SUP-004', 'TechParts Inc',           'US-EIN-98-7654321', 'active',   NULL,      'US',  TRUE),
  ('SUP-005', 'TechParts India Pvt Ltd', 'IN-GSTIN-29ABCDE',  'active',   'SUP-004', 'IN',  FALSE),
  ('SUP-006', 'OfficeWorks Ltd',         'GB-VAT-123456789',  'active',   NULL,      'GB',  TRUE),
  ('SUP-007', 'Furniture Direct Co',     NULL,                 'active',   NULL,      'US',  FALSE),
  ('SUP-008', 'CleanServ LLC',           'US-EIN-55-1234567', 'inactive', NULL,      'US',  FALSE),
  ('SUP-009', 'GlobalLogistics SA',      'CH-VAT-CHE-123',    'active',   NULL,      'CH',  TRUE),
  ('SUP-010', 'Patel & Sons Trading',    'IN-GSTIN-07XYZAB',  'blocked',  NULL,      'IN',  FALSE);

-- ============================================================
-- Commodity — hierarchical classification (3 levels)
-- ============================================================

CREATE TABLE "Commodity" (
  "commodityId"        VARCHAR(20)  PRIMARY KEY,
  "code"               VARCHAR(20)  NOT NULL,
  "name"               VARCHAR(120) NOT NULL,
  "level"              INTEGER      NOT NULL,
  "parentCommodityId"  VARCHAR(20)            -- self-ref for hierarchy
);

INSERT INTO "Commodity" VALUES
  -- Level 1: top categories
  ('COM-100', '43000000', 'IT Equipment',            1, NULL),
  ('COM-200', '44000000', 'Office Supplies',          1, NULL),
  ('COM-300', '76000000', 'Facilities & Maintenance', 1, NULL),
  ('COM-400', '78000000', 'Logistics & Transport',    1, NULL),
  -- Level 2: sub-categories
  ('COM-110', '43210000', 'IT Hardware',              2, 'COM-100'),
  ('COM-120', '43230000', 'IT Software',              2, 'COM-100'),
  ('COM-130', '43240000', 'IT Services',              2, 'COM-100'),
  ('COM-210', '44120000', 'Office Furniture',          2, 'COM-200'),
  ('COM-220', '44110000', 'Office Consumables',        2, 'COM-200'),
  ('COM-310', '76110000', 'Cleaning Services',         2, 'COM-300'),
  ('COM-410', '78100000', 'Freight',                   2, 'COM-400'),
  -- Level 3: specific items
  ('COM-111', '43211500', 'Laptops & Notebooks',       3, 'COM-110'),
  ('COM-112', '43211600', 'Keyboards & Peripherals',   3, 'COM-110'),
  ('COM-113', '43211700', 'Monitors & Displays',       3, 'COM-110'),
  ('COM-121', '43231500', 'Enterprise Software',       3, 'COM-120'),
  ('COM-211', '44121600', 'Ergonomic Chairs',          3, 'COM-210'),
  ('COM-212', '44121700', 'Desks & Workstations',      3, 'COM-210');

-- ============================================================
-- FXRate — currency conversion feed
-- ============================================================

CREATE TABLE "FXRate" (
  "rateId"          VARCHAR(20)  PRIMARY KEY,
  "fromCurrency"    VARCHAR(3)   NOT NULL,
  "toCurrency"      VARCHAR(3)   NOT NULL,
  "rate"            DECIMAL(12,6) NOT NULL,
  "effectiveDate"   DATE         NOT NULL
);

INSERT INTO "FXRate" VALUES
  ('FX-001', 'EUR', 'USD', 1.085000, '2026-04-01'),
  ('FX-002', 'EUR', 'USD', 1.092000, '2026-05-01'),
  ('FX-003', 'EUR', 'USD', 1.078000, '2026-06-01'),
  ('FX-004', 'GBP', 'USD', 1.263000, '2026-04-01'),
  ('FX-005', 'GBP', 'USD', 1.271000, '2026-05-01'),
  ('FX-006', 'GBP', 'USD', 1.258000, '2026-06-01'),
  ('FX-007', 'INR', 'USD', 0.011900, '2026-04-01'),
  ('FX-008', 'INR', 'USD', 0.011850, '2026-05-01'),
  ('FX-009', 'INR', 'USD', 0.011920, '2026-06-01'),
  ('FX-010', 'CHF', 'USD', 1.105000, '2026-04-01'),
  ('FX-011', 'SGD', 'USD', 0.745000, '2026-04-01'),
  ('FX-012', 'USD', 'USD', 1.000000, '2026-01-01');

-- ============================================================
-- CostCenter
-- ============================================================

CREATE TABLE "CostCenter" (
  "costCenterId"  VARCHAR(20)  PRIMARY KEY,
  "name"          VARCHAR(120) NOT NULL,
  "department"    VARCHAR(80)  NOT NULL,
  "region"        VARCHAR(40)  NOT NULL
);

INSERT INTO "CostCenter" VALUES
  ('CC-IT-US',   'IT Operations US',     'Information Technology', 'US'),
  ('CC-IT-EMEA', 'IT Operations EMEA',   'Information Technology', 'EMEA'),
  ('CC-IT-APAC', 'IT Operations APAC',   'Information Technology', 'APAC'),
  ('CC-FAC-US',  'Facilities US',        'Facilities',            'US'),
  ('CC-FIN-US',  'Finance US',           'Finance',               'US'),
  ('CC-PROC-GL', 'Procurement Global',   'Procurement',           'Global');

-- ============================================================
-- Budget
-- ============================================================

CREATE TABLE "Budget" (
  "budgetId"        VARCHAR(20)    PRIMARY KEY,
  "costCenterId"    VARCHAR(20)    NOT NULL,
  "period"          VARCHAR(10)    NOT NULL,  -- e.g. '2026-Q2'
  "allocatedAmount" DECIMAL(14,2)  NOT NULL,
  "currency"        VARCHAR(3)     NOT NULL
);

INSERT INTO "Budget" VALUES
  ('BUD-001', 'CC-IT-US',   '2026-Q2', 750000.00,  'USD'),
  ('BUD-002', 'CC-IT-EMEA', '2026-Q2', 620000.00,  'EUR'),
  ('BUD-003', 'CC-IT-APAC', '2026-Q2', 450000.00,  'SGD'),
  ('BUD-004', 'CC-FAC-US',  '2026-Q2', 180000.00,  'USD'),
  ('BUD-005', 'CC-FIN-US',  '2026-Q2', 95000.00,   'USD'),
  ('BUD-006', 'CC-PROC-GL', '2026-Q2', 320000.00,  'USD');

-- ============================================================
-- Contract
-- ============================================================

CREATE TABLE "Contract" (
  "contractId"   VARCHAR(20)   PRIMARY KEY,
  "title"        VARCHAR(200)  NOT NULL,
  "startDate"    DATE          NOT NULL,
  "endDate"      DATE          NOT NULL,
  "totalValue"   DECIMAL(14,2) NOT NULL,
  "currency"     VARCHAR(3)    NOT NULL,
  "status"       VARCHAR(20)   NOT NULL,   -- draft | active | expired
  "supplierId"   VARCHAR(20)   NOT NULL
);

INSERT INTO "Contract" VALUES
  ('CTR-001', 'IT Hardware Master — Acme Global',       '2025-01-01', '2027-12-31', 2400000.00, 'USD', 'active',  'SUP-001'),
  ('CTR-002', 'IT Hardware EMEA — Acme EMEA',           '2025-06-01', '2027-05-31', 850000.00,  'EUR', 'active',  'SUP-002'),
  ('CTR-003', 'Office Furniture — OfficeWorks',          '2025-03-01', '2026-08-31', 320000.00,  'GBP', 'active',  'SUP-006'),
  ('CTR-004', 'IT Peripherals — TechParts',              '2026-01-01', '2027-12-31', 600000.00,  'USD', 'active',  'SUP-004'),
  ('CTR-005', 'Cleaning Services — CleanServ',           '2024-01-01', '2025-12-31', 120000.00,  'USD', 'expired', 'SUP-008'),
  ('CTR-006', 'Logistics — GlobalLogistics',             '2025-07-01', '2027-06-30', 980000.00,  'CHF', 'active',  'SUP-009'),
  ('CTR-007', 'IT Hardware APAC — Acme APAC',            '2025-09-01', '2027-08-31', 550000.00,  'SGD', 'active',  'SUP-003');

-- ============================================================
-- ContractScope — region, entity, commodity constraints
-- ============================================================

CREATE TABLE "ContractScope" (
  "scopeId"         VARCHAR(20)  PRIMARY KEY,
  "contractId"      VARCHAR(20)  NOT NULL,
  "region"          VARCHAR(40)  NOT NULL,
  "legalEntity"     VARCHAR(80)  NOT NULL,
  "commodityGroup"  VARCHAR(20)  NOT NULL,     -- references Commodity.commodityId
  "validFrom"       DATE         NOT NULL,
  "validTo"         DATE         NOT NULL,
  "exclusions"      VARCHAR(200)               -- free-text exclusions
);

INSERT INTO "ContractScope" VALUES
  ('SCP-001', 'CTR-001', 'Global',      'Corp HQ',           'COM-110', '2025-01-01', '2027-12-31', NULL),
  ('SCP-002', 'CTR-002', 'EMEA',        'Acme EMEA GmbH',   'COM-110', '2025-06-01', '2027-05-31', 'Excludes India subsidiaries'),
  ('SCP-003', 'CTR-003', 'UK',          'OfficeWorks UK',    'COM-210', '2025-03-01', '2026-08-31', NULL),
  ('SCP-004', 'CTR-004', 'US',          'TechParts HQ',      'COM-112', '2026-01-01', '2027-12-31', NULL),
  ('SCP-005', 'CTR-004', 'APAC',        'TechParts HQ',      'COM-112', '2026-01-01', '2027-12-31', 'Excludes Japan'),
  ('SCP-006', 'CTR-006', 'Global',      'GlobalLogistics SA', 'COM-410', '2025-07-01', '2027-06-30', NULL),
  ('SCP-007', 'CTR-007', 'APAC',        'Acme APAC Pte Ltd', 'COM-110', '2025-09-01', '2027-08-31', NULL),
  ('SCP-008', 'CTR-001', 'US',          'Corp HQ',           'COM-120', '2025-01-01', '2027-12-31', NULL);

-- ============================================================
-- PurchaseOrder
-- ============================================================

CREATE TABLE "PurchaseOrder" (
  "poId"         VARCHAR(20)    PRIMARY KEY,
  "amount"       DECIMAL(12,2)  NOT NULL,
  "currency"     VARCHAR(3)     NOT NULL,
  "status"       VARCHAR(20)    NOT NULL,   -- open | received | closed
  "supplierId"   VARCHAR(20)    NOT NULL,
  "orderDate"    DATE           NOT NULL
);

INSERT INTO "PurchaseOrder" VALUES
  ('PO-1001', 48000.00,  'USD', 'received', 'SUP-001', '2026-03-15'),
  ('PO-1002', 22500.00,  'EUR', 'received', 'SUP-002', '2026-04-02'),
  ('PO-1003', 8400.00,   'GBP', 'received', 'SUP-006', '2026-04-10'),
  ('PO-1004', 15600.00,  'USD', 'received', 'SUP-004', '2026-04-18'),
  ('PO-1005', 62000.00,  'USD', 'open',     'SUP-001', '2026-05-01'),
  ('PO-1006', 3200.00,   'INR', 'received', 'SUP-005', '2026-05-05'),
  ('PO-1007', 124500.00, 'CHF', 'open',     'SUP-009', '2026-05-12'),
  ('PO-1008', 9800.00,   'USD', 'received', 'SUP-007', '2026-04-25'),
  ('PO-1009', 35000.00,  'SGD', 'received', 'SUP-003', '2026-05-20'),
  ('PO-1010', 18000.00,  'EUR', 'received', 'SUP-002', '2026-06-01');

-- ============================================================
-- Invoice
-- ============================================================

CREATE TABLE "Invoice" (
  "invoiceId"    VARCHAR(20)    PRIMARY KEY,
  "invoiceDate"  DATE           NOT NULL,
  "amount"       DECIMAL(12,2)  NOT NULL,
  "currency"     VARCHAR(3)     NOT NULL,
  "status"       VARCHAR(20)    NOT NULL,   -- pending | matched | paid | disputed | exception
  "supplierId"   VARCHAR(20)    NOT NULL,
  "poId"         VARCHAR(20)                -- nullable for non-PO invoices
);

INSERT INTO "Invoice" VALUES
  ('INV-2001', '2026-04-20', 48000.00,  'USD', 'paid',      'SUP-001', 'PO-1001'),
  ('INV-2002', '2026-04-28', 23175.00,  'EUR', 'exception', 'SUP-002', 'PO-1002'),  -- 3% over PO
  ('INV-2003', '2026-05-05', 8400.00,   'GBP', 'paid',      'SUP-006', 'PO-1003'),
  ('INV-2004', '2026-05-10', 16224.00,  'USD', 'exception', 'SUP-004', 'PO-1004'),  -- 4% over PO
  ('INV-2005', '2026-05-18', 62000.00,  'USD', 'matched',   'SUP-001', 'PO-1005'),
  ('INV-2006', '2026-05-22', 3200.00,   'INR', 'paid',      'SUP-005', 'PO-1006'),
  ('INV-2007', '2026-06-01', 9800.00,   'USD', 'paid',      'SUP-007', 'PO-1008'),  -- off-contract (no contract with SUP-007)
  ('INV-2008', '2026-06-05', 36050.00,  'SGD', 'exception', 'SUP-003', 'PO-1009'),  -- 3% over PO
  ('INV-2009', '2026-06-10', 18000.00,  'EUR', 'matched',   'SUP-002', 'PO-1010'),
  ('INV-2010', '2026-06-12', 5500.00,   'USD', 'pending',   'SUP-010', NULL),        -- blocked supplier, no PO
  ('INV-2011', '2026-04-22', 48000.00,  'USD', 'paid',      'SUP-001', 'PO-1001'),  -- DUPLICATE of INV-2001
  ('INV-2012', '2026-05-15', 4200.00,   'USD', 'paid',      'SUP-007', NULL);        -- maverick, no PO

-- ============================================================
-- Item — line items on PRs, POs, and invoices
-- ============================================================

CREATE TABLE "Item" (
  "itemId"       VARCHAR(20)    PRIMARY KEY,
  "description"  VARCHAR(200)   NOT NULL,
  "quantity"     DECIMAL(10,2)  NOT NULL,
  "unitPrice"    DECIMAL(10,2)  NOT NULL,
  "currency"     VARCHAR(3)     NOT NULL,
  "commodityId"  VARCHAR(20)    NOT NULL,
  "prId"         VARCHAR(20),
  "poId"         VARCHAR(20),
  "invoiceId"    VARCHAR(20)
);

INSERT INTO "Item" VALUES
  ('ITM-001', 'Dell Latitude 5540 Laptop',        40,  1200.00, 'USD', 'COM-111', NULL,     'PO-1001', 'INV-2001'),
  ('ITM-002', 'Logitech MX Keys Keyboard',       150,  150.00,  'EUR', 'COM-112', NULL,     'PO-1002', 'INV-2002'),
  ('ITM-003', 'Herman Miller Aeron Chair',         12,  700.00,  'GBP', 'COM-211', NULL,     'PO-1003', 'INV-2003'),
  ('ITM-004', 'Dell UltraSharp 27" Monitor',       60,  260.00,  'USD', 'COM-113', NULL,     'PO-1004', 'INV-2004'),
  ('ITM-005', 'ThinkPad X1 Carbon Gen 11',         40,  1550.00, 'USD', 'COM-111', NULL,     'PO-1005', 'INV-2005'),
  ('ITM-006', 'USB-C Hub (generic)',               200,  16.00,   'INR', 'COM-112', NULL,     'PO-1006', 'INV-2006'),
  ('ITM-007', 'Standing Desk Converter',            20,  490.00,  'USD', 'COM-212', NULL,     'PO-1008', 'INV-2007'),
  ('ITM-008', 'MacBook Pro 14"',                    20,  1750.00, 'SGD', 'COM-111', NULL,     'PO-1009', 'INV-2008'),
  ('ITM-009', 'Logitech MX Master Mouse',          100,  180.00,  'EUR', 'COM-112', NULL,     'PO-1010', 'INV-2009'),
  ('ITM-010', 'Whiteboard markers bulk',            50,  110.00,  'USD', 'COM-220', NULL,     NULL,      'INV-2010'),
  ('ITM-011', 'Dell Latitude 5540 Laptop (dup)',    40, 1200.00,  'USD', 'COM-111', NULL,     'PO-1001', 'INV-2011'),
  ('ITM-012', 'Miscellaneous office supplies',      1,  4200.00,  'USD', 'COM-220', NULL,     NULL,      'INV-2012');

-- ============================================================
-- Requisition
-- ============================================================

CREATE TABLE "Requisition" (
  "prId"          VARCHAR(20)  PRIMARY KEY,
  "description"   VARCHAR(300) NOT NULL,
  "requester"     VARCHAR(80)  NOT NULL,
  "location"      VARCHAR(80),
  "quantity"       INTEGER,
  "costCenterId"  VARCHAR(20),
  "status"        VARCHAR(20)  NOT NULL    -- draft | submitted | approved | fulfilled
);

INSERT INTO "Requisition" VALUES
  ('PR-3001', '500 ergonomic keyboards for Bangalore office',     'Priya Sharma',   'Bangalore', 500,  'CC-IT-APAC', 'submitted'),
  ('PR-3002', 'Replace 30 aging laptops in NYC finance team',      'Mike Chen',       'New York',  30,   'CC-FIN-US',  'approved'),
  ('PR-3003', 'Standing desks for new hires, London office',       'Sarah Jones',     'London',    15,   'CC-FAC-US',  'draft'),
  ('PR-3004', 'Annual software license renewal — enterprise suite', 'IT Procurement', NULL,        1,    'CC-IT-US',   'approved');

-- ============================================================
-- ToleranceException — flagged invoice/PO variances
-- ============================================================

CREATE TABLE "ToleranceException" (
  "exceptionId"    VARCHAR(20)    PRIMARY KEY,
  "invoiceId"      VARCHAR(20)    NOT NULL,
  "poId"           VARCHAR(20)    NOT NULL,
  "exceptionType"  VARCHAR(30)    NOT NULL,  -- price_variance | quantity_variance | tax_variance
  "thresholdPct"   DECIMAL(5,2)   NOT NULL,
  "actualPct"      DECIMAL(5,2)   NOT NULL,
  "status"         VARCHAR(30)    NOT NULL   -- open | auto_resolved | manually_resolved | escalated
);

INSERT INTO "ToleranceException" VALUES
  ('EXC-001', 'INV-2002', 'PO-1002', 'price_variance',    3.00,  3.00, 'open'),
  ('EXC-002', 'INV-2004', 'PO-1004', 'price_variance',    3.00,  4.00, 'open'),
  ('EXC-003', 'INV-2008', 'PO-1009', 'price_variance',    3.00,  3.00, 'open'),
  ('EXC-004', 'INV-2002', 'PO-1002', 'price_variance',    3.00,  3.00, 'auto_resolved'),  -- historical duplicate
  ('EXC-005', 'INV-2004', 'PO-1004', 'quantity_variance',  2.00,  0.00, 'auto_resolved');  -- quantity matched, price didn't

-- ============================================================
-- ExceptionRule — auto-resolution rules
-- ============================================================

CREATE TABLE "ExceptionRule" (
  "ruleId"       VARCHAR(20)  PRIMARY KEY,
  "commodityId"  VARCHAR(20)  NOT NULL,
  "supplierId"   VARCHAR(20),               -- NULL = applies to all suppliers
  "condition"    VARCHAR(300) NOT NULL,
  "resolution"   VARCHAR(200) NOT NULL,
  "approvedBy"   VARCHAR(80)  NOT NULL,
  "createdDate"  DATE         NOT NULL
);

INSERT INTO "ExceptionRule" VALUES
  ('RULE-001', 'COM-112', 'SUP-002', 'price_variance <= 5% AND supplier.preferredFlag = true',
   'auto_approve', 'AP Manager (J. Williams)', '2026-01-15'),
  ('RULE-002', 'COM-111', NULL,       'price_variance <= 3% AND quantity_variance = 0',
   'auto_approve', 'AP Manager (J. Williams)', '2026-02-01'),
  ('RULE-003', 'COM-211', 'SUP-006', 'price_variance <= 2%',
   'auto_approve_with_note', 'Facilities Director',    '2025-11-20');

-- ============================================================
-- SpendRecord — normalized, currency-converted spend facts
-- ============================================================

CREATE TABLE "SpendRecord" (
  "spendId"           VARCHAR(20)    PRIMARY KEY,
  "invoiceId"         VARCHAR(20)    NOT NULL,
  "supplierId"        VARCHAR(20)    NOT NULL,
  "commodityId"       VARCHAR(20)    NOT NULL,
  "rateId"            VARCHAR(20),                 -- NULL if already USD
  "originalAmount"    DECIMAL(12,2)  NOT NULL,
  "originalCurrency"  VARCHAR(3)     NOT NULL,
  "normalizedAmount"  DECIMAL(12,2)  NOT NULL,     -- in USD
  "period"            VARCHAR(10)    NOT NULL,      -- e.g. '2026-Q2'
  "onContract"        BOOLEAN        NOT NULL
);

INSERT INTO "SpendRecord" VALUES
  ('SPD-001', 'INV-2001', 'SUP-001', 'COM-111', 'FX-012', 48000.00, 'USD', 48000.00,  '2026-Q2', TRUE),
  ('SPD-002', 'INV-2002', 'SUP-002', 'COM-112', 'FX-001', 23175.00, 'EUR', 25144.88,  '2026-Q2', TRUE),
  ('SPD-003', 'INV-2003', 'SUP-006', 'COM-211', 'FX-004', 8400.00,  'GBP', 10609.20,  '2026-Q2', TRUE),
  ('SPD-004', 'INV-2004', 'SUP-004', 'COM-113', 'FX-012', 16224.00, 'USD', 16224.00,  '2026-Q2', TRUE),
  ('SPD-005', 'INV-2005', 'SUP-001', 'COM-111', 'FX-012', 62000.00, 'USD', 62000.00,  '2026-Q2', TRUE),
  ('SPD-006', 'INV-2006', 'SUP-005', 'COM-112', 'FX-007', 3200.00,  'INR', 38.08,     '2026-Q2', FALSE),  -- SUP-005 not preferred, no contract
  ('SPD-007', 'INV-2007', 'SUP-007', 'COM-212', 'FX-012', 9800.00,  'USD', 9800.00,   '2026-Q2', FALSE),  -- no contract with SUP-007
  ('SPD-008', 'INV-2008', 'SUP-003', 'COM-111', 'FX-011', 36050.00, 'SGD', 26857.25,  '2026-Q2', TRUE),
  ('SPD-009', 'INV-2009', 'SUP-002', 'COM-112', 'FX-003', 18000.00, 'EUR', 19404.00,  '2026-Q2', TRUE),
  ('SPD-010', 'INV-2010', 'SUP-010', 'COM-220', 'FX-012', 5500.00,  'USD', 5500.00,   '2026-Q2', FALSE),  -- blocked supplier
  ('SPD-011', 'INV-2011', 'SUP-001', 'COM-111', 'FX-012', 48000.00, 'USD', 48000.00,  '2026-Q2', TRUE),   -- DUPLICATE spend
  ('SPD-012', 'INV-2012', 'SUP-007', 'COM-220', 'FX-012', 4200.00,  'USD', 4200.00,   '2026-Q2', FALSE);  -- maverick, no PO
