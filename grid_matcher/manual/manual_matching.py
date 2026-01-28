#!/usr/bin/env python
# manual_matching.py - Module for manual matching of JAO and PyPSA lines

import os
import json
import pandas as pd
from pathlib import Path


def add_predefined_manual_matches(jao_gdf, pypsa_gdf, manual_matches=None, interactive=False):
    """
    Add predefined manual matches plus allow for interactive additions.

    Parameters:
    -----------
    jao_gdf : GeoDataFrame
        JAO lines
    pypsa_gdf : GeoDataFrame
        PyPSA lines
    manual_matches : list, optional
        Existing manual matches to update
    interactive : bool, optional
        Whether to enable interactive mode for adding additional matches

    Returns:
    --------
    list
        Updated manual matches
    """
    if manual_matches is None:
        manual_matches = []

    print("\n===== ADDING PREDEFINED MANUAL MATCHES =====")

    # Dictionary of predefined matches (JAO ID -> PyPSA IDs)
    predefined_matches = {
        # Format: 'jao_id': ['pypsa_id1', 'pypsa_id2', ...]
        '2611': ['merged_relation/3916226-380-c+3', 'merged_way/240543053-1-380-a+5'],
        '2695': ['merged_way/30970013-1-380+3'],
        '2908': ['relation/3717719-220'],
        '2566': ['relation/3918229-220-b'],
        '2916': ['merged_way/27854188-2-220+1'],
        '2719': ['merged_way/30970013-2-220+3'],
        '390': ['merged_way/26565859-380+1'],
        '392': ['merged_way/26565859-380+1'],
        '2488': ['way/45737386-380'],
        '2465': ['merged_way/26971631-220+1'],
        '2629': ['merged_way/156350633-220+3'],
        '2498': ['merged_way/296751942-220+1'],
        '2509': ['merged_way/178708511-220+1'],
        '2875': ['merged_way/114397559-380+3'],
        '2557': ['merged_way/842418530-1-380+1', 'way/41232147-380'],
        '2612': ['merged_relation/3989163-380-e+2'],
        '2618': ['merged_relation/14514288-380-b+5'],
        '245': ['merged_way/1104699060-380+2', 'way/108526453-380', 'way/108526459-1-380', 'way/95860782-380'],



    }

    # Track which matches were successfully added
    added_matches = []
    skipped_matches = []

    # Process predefined matches
    for jao_id, pypsa_ids in predefined_matches.items():
        print(f"\nProcessing predefined match: JAO {jao_id} -> {', '.join(pypsa_ids)}")

        # Validate JAO ID
        if not validate_jao_id(jao_id, jao_gdf):
            print(f"Error: JAO ID '{jao_id}' not found in dataset")
            skipped_matches.append((jao_id, pypsa_ids, "JAO ID not found"))
            continue

        # Validate PyPSA IDs
        valid_pypsa_ids, message = validate_pypsa_ids(pypsa_ids, pypsa_gdf)

        if message:
            print(f"Warning: {message}")

        if not valid_pypsa_ids:
            print(f"Error: No valid PyPSA IDs for match with JAO {jao_id}")
            skipped_matches.append((jao_id, pypsa_ids, "No valid PyPSA IDs"))
            continue

        # Check if this JAO ID already has a manual match
        existing_match_index = None
        for i, match in enumerate(manual_matches):
            if match.get('jao_id') == jao_id:
                existing_match_index = i
                break

        # Create the new match
        new_match = {
            'jao_id': jao_id,
            'pypsa_ids': valid_pypsa_ids,
            'note': "Predefined manual match"
        }

        # Add to manual matches (or replace existing)
        if existing_match_index is not None:
            print(f"Replacing existing match for JAO {jao_id}")
            manual_matches[existing_match_index] = new_match
        else:
            print(f"Adding new match for JAO {jao_id}")
            manual_matches.append(new_match)

        added_matches.append((jao_id, valid_pypsa_ids))

    # Summary of predefined matches
    print("\n----- Predefined Matches Summary -----")
    print(f"Successfully added/updated {len(added_matches)} predefined matches:")
    for jao_id, pypsa_ids in added_matches:
        print(f"  JAO {jao_id} -> {', '.join(pypsa_ids)}")

    if skipped_matches:
        print(f"\nSkipped {len(skipped_matches)} predefined matches:")
        for jao_id, pypsa_ids, reason in skipped_matches:
            print(f"  JAO {jao_id} -> {', '.join(pypsa_ids)} (Reason: {reason})")

    # Only show interactive prompt if interactive mode is enabled
    if interactive:
        # Ask if user wants to add additional manual matches
        while True:
            add_more = input("\nDo you want to add more manual matches interactively? (y/n): ").strip().lower()
            if add_more != 'y':
                break

            # Get JAO ID
            jao_id = input("Enter JAO ID: ").strip()

            # Validate JAO ID
            if not validate_jao_id(jao_id, jao_gdf):
                print(f"Error: JAO ID '{jao_id}' not found in dataset")
                continue

            # Get PyPSA IDs
            pypsa_ids_input = input("Enter PyPSA IDs (comma-separated): ").strip()
            pypsa_ids = [pid.strip() for pid in pypsa_ids_input.split(',') if pid.strip()]

            # Validate PyPSA IDs
            valid_pypsa_ids, message = validate_pypsa_ids(pypsa_ids, pypsa_gdf)

            if message:
                print(f"Warning: {message}")

            if not valid_pypsa_ids:
                print(f"Error: No valid PyPSA IDs provided")
                continue

            # Get note
            note = input("Enter note (optional): ").strip()

            # Create match entry
            new_match = {
                'jao_id': jao_id,
                'pypsa_ids': valid_pypsa_ids,
                'note': note or "Manual match"
            }

            # Check for existing match
            existing_match_index = None
            for i, match in enumerate(manual_matches):
                if match.get('jao_id') == jao_id:
                    existing_match_index = i
                    break

            # Add or update
            if existing_match_index is not None:
                manual_matches[existing_match_index] = new_match
                print(f"Updated match for JAO {jao_id}")
            else:
                manual_matches.append(new_match)
                print(f"Added new match for JAO {jao_id}")

    print(f"\nTotal manual matches: {len(manual_matches)}")
    return manual_matches


def import_new_lines_from_csv(jao_gdf, pypsa_gdf, data_dir="grid_matcher/data"):
    """
    Import new JAO and PyPSA lines from CSV files and add them to the existing GeoDataFrames.

    Parameters:
    -----------
    jao_gdf : GeoDataFrame
        Existing JAO lines
    pypsa_gdf : GeoDataFrame
        Existing PyPSA lines
    data_dir : str
        Directory containing the CSV files

    Returns:
    --------
    tuple
        (updated_jao_gdf, updated_pypsa_gdf)
    """
    from pathlib import Path
    import pandas as pd
    import geopandas as gpd

    print("\n===== IMPORTING NEW LINES FROM CSV =====")

    # Make sure data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} doesn't exist")
        return jao_gdf, pypsa_gdf

    # Define paths for new line files
    jao_new_path = data_path / "jao-new-lines.csv"
    pypsa_new_path = data_path / "pypsa-new-lines.csv"

    # Import new JAO lines
    new_jao_gdf = None
    if jao_new_path.exists():
        try:
            # Read CSV file
            jao_new_df = pd.read_csv(jao_new_path)

            # Parse geometry column
            def parse_wkt(wkt_str):
                """Parse WKT geometry string."""
                from shapely import wkt
                try:
                    return wkt.loads(wkt_str)
                except Exception as e:
                    print(f"Error parsing WKT: {e}")
                    return None

            # Convert to GeoDataFrame
            jao_new_geometry = jao_new_df['geometry'].apply(parse_wkt)
            new_jao_gdf = gpd.GeoDataFrame(jao_new_df, geometry=jao_new_geometry)

            print(f"Loaded {len(new_jao_gdf)} new JAO lines")

            # Check for duplicate IDs
            existing_ids = set(jao_gdf['id'].astype(str))
            new_ids = set(new_jao_gdf['id'].astype(str))
            duplicate_ids = existing_ids.intersection(new_ids)

            if duplicate_ids:
                print(f"Warning: Found {len(duplicate_ids)} duplicate JAO IDs: {', '.join(duplicate_ids)}")
                print("Removing duplicates from new data...")
                new_jao_gdf = new_jao_gdf[~new_jao_gdf['id'].astype(str).isin(duplicate_ids)]

            # Check if we have any valid new lines after removing duplicates
            if len(new_jao_gdf) > 0:
                # Ensure same CRS
                if jao_gdf.crs is not None and new_jao_gdf.crs is None:
                    new_jao_gdf.set_crs(jao_gdf.crs, inplace=True)

                # Concatenate with existing data
                updated_jao_gdf = pd.concat([jao_gdf, new_jao_gdf], ignore_index=True)
                print(f"Added {len(new_jao_gdf)} new JAO lines")
            else:
                print("No new unique JAO lines to add")
                updated_jao_gdf = jao_gdf
        except Exception as e:
            print(f"Error importing new JAO lines: {e}")
            updated_jao_gdf = jao_gdf
    else:
        print(f"No new JAO lines file found at {jao_new_path}")
        updated_jao_gdf = jao_gdf

    # Import new PyPSA lines
    new_pypsa_gdf = None
    if pypsa_new_path.exists():
        try:
            # Read CSV file
            pypsa_new_df = pd.read_csv(pypsa_new_path)

            # Convert to GeoDataFrame
            pypsa_new_geometry = pypsa_new_df['geometry'].apply(parse_wkt)
            new_pypsa_gdf = gpd.GeoDataFrame(pypsa_new_df, geometry=pypsa_new_geometry)

            print(f"Loaded {len(new_pypsa_gdf)} new PyPSA lines")

            # Check for duplicate IDs
            id_field = 'line_id' if 'line_id' in pypsa_gdf.columns else 'id'
            existing_ids = set(pypsa_gdf[id_field].astype(str))
            new_ids = set(new_pypsa_gdf[id_field].astype(str))
            duplicate_ids = existing_ids.intersection(new_ids)

            if duplicate_ids:
                print(f"Warning: Found {len(duplicate_ids)} duplicate PyPSA IDs")
                print("Removing duplicates from new data...")
                new_pypsa_gdf = new_pypsa_gdf[~new_pypsa_gdf[id_field].astype(str).isin(duplicate_ids)]

            # Check if we have any valid new lines after removing duplicates
            if len(new_pypsa_gdf) > 0:
                # Ensure same CRS
                if pypsa_gdf.crs is not None and new_pypsa_gdf.crs is None:
                    new_pypsa_gdf.set_crs(pypsa_gdf.crs, inplace=True)

                # Concatenate with existing data
                updated_pypsa_gdf = pd.concat([pypsa_gdf, new_pypsa_gdf], ignore_index=True)
                print(f"Added {len(new_pypsa_gdf)} new PyPSA lines")
            else:
                print("No new unique PyPSA lines to add")
                updated_pypsa_gdf = pypsa_gdf
        except Exception as e:
            print(f"Error importing new PyPSA lines: {e}")
            updated_pypsa_gdf = pypsa_gdf
    else:
        print(f"No new PyPSA lines file found at {pypsa_new_path}")
        updated_pypsa_gdf = pypsa_gdf

    # Summarize the import
    print("\n----- Import Summary -----")
    print(f"Original JAO lines: {len(jao_gdf)}, Updated: {len(updated_jao_gdf)}")
    print(f"Original PyPSA lines: {len(pypsa_gdf)}, Updated: {len(updated_pypsa_gdf)}")

    # Return the updated GeoDataFrames
    return updated_jao_gdf, updated_pypsa_gdf

def load_manual_matches_file(file_path):
    """Load manual matches from a JSON file if it exists."""
    if not os.path.exists(file_path):
        print(f"Manual matches file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r') as f:
            matches = json.load(f)
        print(f"Loaded {len(matches)} manual matches from {file_path}")
        return matches
    except Exception as e:
        print(f"Error loading manual matches file: {e}")
        return []


def save_manual_matches_file(matches, file_path):
    """Save manual matches to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(matches, f, indent=2)
        print(f"Saved {len(matches)} manual matches to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving manual matches file: {e}")
        return False


def validate_jao_id(jao_id, jao_gdf):
    """Validate that a JAO ID exists in the dataset."""
    if jao_id is None or jao_id == '':
        return False

    jao_rows = jao_gdf[jao_gdf['id'].astype(str) == str(jao_id)]
    return not jao_rows.empty


def validate_pypsa_ids(pypsa_ids, pypsa_gdf):
    """Validate that PyPSA IDs exist in the dataset."""
    if not pypsa_ids:
        return [], "No PyPSA IDs provided"

    valid_ids = []
    invalid_ids = []

    for pypsa_id in pypsa_ids:
        if pypsa_id is None or pypsa_id == '':
            invalid_ids.append(pypsa_id)
            continue

        pypsa_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == str(pypsa_id)]
        if pypsa_rows.empty and 'line_id' in pypsa_gdf.columns:
            pypsa_rows = pypsa_gdf[pypsa_gdf['line_id'].astype(str) == str(pypsa_id)]

        if pypsa_rows.empty:
            invalid_ids.append(pypsa_id)
        else:
            valid_ids.append(str(pypsa_id))

    message = ""
    if invalid_ids:
        message = f"Invalid PyPSA IDs: {', '.join(invalid_ids)}"

    return valid_ids, message





def apply_manual_matches(results, jao_gdf, pypsa_gdf, manual_matches):
    """Apply manual matches to the results, adding or updating matches with proper parameter allocation."""
    print("\n===== APPLYING MANUAL MATCHES =====")

    if not manual_matches:
        print("No manual matches to apply")
        return results

    # Create a lookup of existing results by JAO ID
    results_by_jao = {str(result.get('jao_id')): result for result in results}

    # Track statistics
    added = 0
    updated = 0
    skipped = 0

    for match in manual_matches:
        jao_id = str(match.get('jao_id', ''))
        pypsa_ids = match.get('pypsa_ids', [])
        note = match.get('note', 'Manual match')

        if not jao_id or not pypsa_ids:
            print(f"Skipping invalid manual match: {match}")
            skipped += 1
            continue

        # Get JAO data
        jao_rows = jao_gdf[jao_gdf['id'].astype(str) == jao_id]
        if jao_rows.empty:
            print(f"Warning: JAO ID {jao_id} not found in dataset")
            skipped += 1
            continue

        jao_row = jao_rows.iloc[0]

        # Extract JAO parameters
        jao_name = str(jao_row.get('NE_name', ''))
        jao_voltage = int(jao_row.get('v_nom', jao_row.get('voltage', 0)) or 0)
        jao_r = float(jao_row.get('r', 0) or 0)
        jao_x = float(jao_row.get('x', 0) or 0)
        jao_b = float(jao_row.get('b', 0) or 0)
        jao_length_km = float(jao_row.get('length_km', jao_row.get('length', 0)) or 0)

        # Validate PyPSA IDs
        valid_pypsa_ids, _ = validate_pypsa_ids(pypsa_ids, pypsa_gdf)

        if not valid_pypsa_ids:
            print(f"Skipping manual match for JAO {jao_id}: No valid PyPSA IDs")
            skipped += 1
            continue

        # Create matched_lines_data for parameter allocation
        matched_lines_data = []
        total_pypsa_length_km = 0

        for pypsa_id in valid_pypsa_ids:
            pypsa_rows = pypsa_gdf[pypsa_gdf['id'].astype(str) == str(pypsa_id)]
            if pypsa_rows.empty and 'line_id' in pypsa_gdf.columns:
                pypsa_rows = pypsa_gdf[pypsa_gdf['line_id'].astype(str) == str(pypsa_id)]

            pypsa_row = pypsa_rows.iloc[0]
            length = float(pypsa_row.get('length', 0) or 0)
            circuits = int(pypsa_row.get('circuits', 1) or 1)

            # Convert length to km if in meters
            length_km = length / 1000 if length > 1000 else length
            total_pypsa_length_km += length_km

            # Get original parameters
            original_r = float(pypsa_row.get('r', 0) or 0)
            original_x = float(pypsa_row.get('x', 0) or 0)
            original_b = float(pypsa_row.get('b', 0) or 0)

            matched_line = {
                'network_id': pypsa_id,
                'length_km': length,
                'num_parallel': circuits,
                'allocation_status': 'Manual Match',
                'original_r': original_r,
                'original_x': original_x,
                'original_b': original_b,
            }
            matched_lines_data.append(matched_line)

        # Calculate per-km values for parameter allocation
        if jao_length_km > 0:
            jao_r_per_km = jao_r / jao_length_km
            jao_x_per_km = jao_x / jao_length_km
            jao_b_per_km = jao_b / jao_length_km
        else:
            jao_r_per_km = jao_x_per_km = jao_b_per_km = 0

        # Allocate parameters to each line
        for line in matched_lines_data:
            # Store allocated parameters
            line['allocated_r'] = jao_r
            line['allocated_x'] = jao_x
            line['allocated_b'] = jao_b
            line['allocated_r_per_km'] = jao_r_per_km
            line['allocated_x_per_km'] = jao_x_per_km
            line['allocated_b_per_km'] = jao_b_per_km

        # Calculate length ratio for quality assessment
        length_ratio = total_pypsa_length_km / jao_length_km if jao_length_km > 0 else 0

        # Create or update match in results
        if jao_id in results_by_jao:
            # Update existing result
            existing_result = results_by_jao[jao_id]
            existing_result['matched'] = True
            existing_result['pypsa_ids'] = valid_pypsa_ids
            existing_result['matched_lines_data'] = matched_lines_data
            existing_result['match_quality'] = f"Manual Match: {note}"
            existing_result['manual_match'] = True
            existing_result['jao_r'] = jao_r
            existing_result['jao_x'] = jao_x
            existing_result['jao_b'] = jao_b
            existing_result['jao_length_km'] = jao_length_km
            existing_result['length_ratio'] = length_ratio

            updated += 1
            print(f"Updated match for JAO {jao_id} → {len(valid_pypsa_ids)} PyPSA lines")
        else:
            # Create new result
            new_result = {
                'jao_id': jao_id,
                'jao_name': jao_name,
                'jao_voltage': jao_voltage,
                'matched': True,
                'pypsa_ids': valid_pypsa_ids,
                'matched_lines_data': matched_lines_data,
                'match_quality': f"Manual Match: {note}",
                'manual_match': True,
                'jao_r': jao_r,
                'jao_x': jao_x,
                'jao_b': jao_b,
                'jao_length_km': jao_length_km,
                'length_ratio': length_ratio
            }

            results.append(new_result)
            added += 1
            print(f"Added match for JAO {jao_id} → {len(valid_pypsa_ids)} PyPSA lines")

    print(f"Manual matching applied: {added} added, {updated} updated, {skipped} skipped")
    return results