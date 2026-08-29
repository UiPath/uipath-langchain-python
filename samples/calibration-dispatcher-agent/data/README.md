# Data Files

This directory contains sample data files for the Calibration Dispatcher Agent.

## Files

### Schema.json
Data Fabric entity definitions for the four entities used by the agent:
- **Equipment**: Medical devices requiring calibration
- **Clinics**: Healthcare facilities where devices are located
- **Technicians**: Field service technicians who perform calibrations
- **ServiceOrders**: Scheduled calibration visits (created by agent)

**Usage**: Import this schema into UiPath Orchestrator Data Service to create the required entities.

### CSV Files

Sample data for testing and demonstration:

- **devices_for_data_fabric.csv** (20 records)
  - Medical devices (Audiometers and Tympanometers)
  - Includes calibration due dates, priorities, and clinic assignments
  
- **locations.csv** (20 records)
  - Healthcare facilities across 4 Polish cities
  - Includes addresses, coordinates, SLA tiers (24h/48h/72h)
  - Fictitious contact information (names and emails)
  
- **technicians.csv** (5 records)
  - Field service technicians with specializations
  - Home base cities for route optimization
  - Fictitious names and contact information

## Data Privacy

All data has been anonymized:
- ✅ Clinic names are generic (e.g., "Regional Hospital No 1")
- ✅ Contact names are common English names
- ✅ Email addresses use `.example` domain
- ✅ Geographic data (cities, coordinates) is real public information

## Usage

### For Mock Mode Testing
The agent loads CSV files directly when `USE_MOCK_DATA=true` in config.py. No additional setup needed.

### For Production Deployment
Import data into Data Fabric:
- manually via Orchestrator UI: **Data Service > Entities > Import**

## Customization

Feel free to modify the CSV files to:
- Add your own cities and clinic locations
- Adjust device counts and types
- Change technician assignments
- Modify SLA tiers based on your business needs

Just maintain the CSV column structure to ensure compatibility.
