"""
ENZYME COLD/HEAT ADAPTATION PROJECT - IMPROVED VERSION
Critical Residue Analysis with Biopython + Advanced Directed Evolution

IMPROVEMENTS:
1. Adaptive mutation rate (based on distance to target)
2. Diversity preservation (diversity injection)
3. Reduced elitism
4. Improved convergence strategy
5. Multi-scale mutation (small and large changes)

Gokce Ceyda Bilgin - ISEF 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import warnings
import tempfile
import shutil
import argparse
import requests
import base64
import re
import sys

os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
# PyCaret model loading
from pycaret.regression import load_model, predict_model

# Biopython imports
from Bio.PDB import PDBParser, PDBList, DSSP, HSExposureCB
from Bio.PDB.Polypeptide import PPBuilder
from collections import defaultdict

# Protein feature calculation function (using enzyme_feature_lib)
import enzyme_feature_lib as elib

# Suppress warning messages
warnings.filterwarnings("ignore")


# ============================================================================
# REPORT FOLDER CREATION
# ============================================================================

REPORT_DIR = "report"

def create_report_dir():
    """Create report folder"""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    return REPORT_DIR

# ============================================================================
# SECTION 1: PDB FILE DOWNLOAD
# ============================================================================

def uniprot_to_pdb(uniprot_id):
    """Query UniProt API to find associated PDB IDs for a given UniProt accession."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=xref_pdb"
    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            pdb_ids = []
            for ref in data.get("uniProtKBCrossReferences", []):
                if ref["database"] == "PDB":
                    pdb_ids.append(ref["id"])
            return pdb_ids
    except Exception as e:
        print(f"   WARNING: UniProt API query failed: {e}")
    return []

def download_pdb_structure(uniprot_id='E5BBQ3'):
    """
    Resolve UniProt ID to PDB ID via UniProt API, then download PDB structure.
    """
    print("\n" + "="*80)
    print(f"STEP 1: RESOLVING UNIPROT {uniprot_id} -> PDB STRUCTURE")
    print("="*80)

    # 1. Resolve UniProt -> PDB
    print(f"   Querying UniProt API for {uniprot_id}...")
    pdb_ids = uniprot_to_pdb(uniprot_id)

    if not pdb_ids:
        print(f"   ERROR: No PDB entry found for UniProt ID '{uniprot_id}'")
        print("   Please check the UniProt ID and try again.")
        return None, None

    pdb_id = pdb_ids[0]
    if len(pdb_ids) > 1:
        print(f"   Multiple PDB entries found: {', '.join(pdb_ids)}")
    print(f"   Using PDB: {pdb_id}")

    pdb_name = f"{pdb_id}.pdb"
    pdb_path = os.path.join(REPORT_DIR, pdb_name)

    # 2. Check if already exists
    if os.path.exists(pdb_path):
        print(f"   {pdb_name} already exists!")
        return pdb_path, pdb_id

    # Try copying from report folder
    original_pdb = os.path.join("report", pdb_name)
    if os.path.exists(original_pdb):
        shutil.copy(original_pdb, pdb_path)
        print(f"   {pdb_name} copied: {original_pdb} -> {pdb_path}")
        return pdb_path, pdb_id

    # 3. Download from RCSB
    print(f"   Downloading {pdb_id} from RCSB...")
    try:
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, pdir=REPORT_DIR, file_format='pdb')

        # Rename downloaded file
        old_path = os.path.join(REPORT_DIR, f'pdb{pdb_id.lower()}.ent')
        if os.path.exists(old_path):
            os.rename(old_path, pdb_path)
            print(f"   {pdb_name} downloaded successfully!")
            return pdb_path, pdb_id
        elif os.path.exists(pdb_path):
            return pdb_path, pdb_id
    except Exception as e:
        print(f"   ERROR: Could not download PDB: {e}")
        print(f"\n   MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"   1. Go to https://www.rcsb.org/structure/{pdb_id}")
        print(f"   2. Select 'Download Files' -> 'PDB Format'")
        print(f"   3. Save the file as '{pdb_path}'")
        return None, None

    return None, None

# ============================================================================
# SECTION 2: CRITICAL RESIDUE ANALYSIS (BIOPYTHON)
# ============================================================================

class CriticalResidueAnalyzer:
    """
    Identifies critical amino acids for enzyme using Biopython
    """
    
    def __init__(self, pdb_file, catalytic_triad=None, oxyanion_hole=None,
                 substrate_binding=None):
        """
        Args:
            pdb_file: PDB file path (4CG1.pdb)
            catalytic_triad: List of catalytic triad residue positions
            oxyanion_hole: List of oxyanion hole residue positions
            substrate_binding: List of substrate binding residue positions
        """
        print("\n" + "="*80)
        print("STEP 2: IDENTIFYING CRITICAL AMINO ACIDS (BIOPYTHON ANALYSIS)")
        print("="*80)

        self.pdb_file = pdb_file
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('TFC', pdb_file)
        self.model = self.structure[0]
        self.chain = list(self.model.get_chains())[0]  # Get the first chain

        # Known critical positions from literature (for TFC)
        self.catalytic_triad = catalytic_triad if catalytic_triad is not None else [130, 176, 208]
        self.oxyanion_hole = oxyanion_hole if oxyanion_hole is not None else [60, 131, 132]
        self.substrate_binding = substrate_binding if substrate_binding is not None else []

        print(f"PDB structure loaded: {len(list(self.chain.get_residues()))} amino acids")
        print(f"Catalytic triad: {self.catalytic_triad}")
        print(f"Oxyanion hole: {self.oxyanion_hole}")
    
    def calculate_distance_to_active_site(self):
        """
        Calculate distance of each amino acid to the active site (catalytic triad)
        """
        print("\n-> Calculating distances to active site...")
        
        # Get CA atom coordinates of catalytic triad
        active_site_coords = []
        for res_num in self.catalytic_triad:
            try:
                residue = self.chain[res_num]
                ca_atom = residue['CA']
                active_site_coords.append(ca_atom.coord)
            except:
                pass
        
        if not active_site_coords:
            print("  WARNING: Catalytic triad coordinates not found!")
            return {}
        
        distances = {}
        for residue in self.chain:
            if residue.id[0] == ' ':  # Standard amino acids only
                res_num = residue.id[1]
                try:
                    ca_coord = residue['CA'].coord
                    
                    # Calculate distance to each active site atom, take minimum
                    min_dist = min([
                        np.linalg.norm(ca_coord - active_coord)
                        for active_coord in active_site_coords
                    ])
                    
                    distances[res_num] = min_dist
                except:
                    pass
        
        print(f"  Distance calculated for {len(distances)} amino acids")
        return distances
    
    def calculate_solvent_accessibility(self):
        """
        Calculate solvent accessibility for each amino acid
        Uses DSSP (must be installed: sudo apt-get install dssp)
        """
        print("\n-> Calculating solvent accessibility...")
        
        try:
            # Calculate RSA with DSSP
            dssp = DSSP(self.model, self.pdb_file, dssp='mkdssp')
            
            accessibility = {}
            for key in dssp:
                res_id = key[1][1]
                rsa = dssp[key][3]  # Relative Solvent Accessibility
                
                # RSA > 0.2 = surface, < 0.2 = buried (core)
                classification = 'surface' if rsa > 0.2 else 'buried'
                accessibility[res_id] = (rsa, classification)
            
            print(f"  {len(accessibility)} amino acids analyzed with DSSP")
            return accessibility
            
        except Exception as e:
            print(f"  WARNING: DSSP failed ({e}), using alternative method...")
            return self._calculate_hse()
    
    def _calculate_hse(self):
        """
        Alternative: Half-Sphere Exposure (if DSSP unavailable)
        """
        try:
            hse = HSExposureCB(self.model)
            accessibility = {}
            
            for residue in self.chain:
                if residue.id[0] == ' ':
                    res_num = residue.id[1]
                    try:
                        hse_up, hse_down, angle = hse[self.chain.id, residue.id]
                        total_hse = hse_up + hse_down
                        
                        # HSE < 30 generally means surface (relaxed threshold)
                        classification = 'surface' if total_hse < 30 else 'buried'
                        accessibility[res_num] = (total_hse, classification)
                    except:
                        pass
            
            print(f"  {len(accessibility)} amino acids analyzed with HSE")
            return accessibility
        except:
            print("  WARNING: HSE also failed, default values will be used")
            return {}
    
    def classify_residues(self):
        """
        Classify each amino acid by criticality level
        """
        print("\n-> Classifying amino acids...")
        
        # Calculate distance and accessibility
        distances = self.calculate_distance_to_active_site()
        accessibility = self.calculate_solvent_accessibility()
        
        classification = {}
        
        for residue in self.chain:
            if residue.id[0] == ' ':
                res_num = residue.id[1]
                res_name = residue.get_resname()
                
                # Default category
                category = 'flexible'
                
                # 1. Catalytic triad and oxyanion hole -> FIXED
                if res_num in self.catalytic_triad or res_num in self.oxyanion_hole or res_num in self.substrate_binding:
                    category = 'fixed'
                
                # 2. Close to active site (< 8 A) -> FIXED
                elif res_num in distances and distances[res_num] < 8.0:
                    category = 'fixed'
                
                # 3. In core (buried) and at medium distance (8-15 A) -> MODERATE
                elif res_num in accessibility and accessibility[res_num][1] == 'buried':
                    if res_num in distances and 8.0 <= distances[res_num] <= 15.0:
                        category = 'moderate'
                    else:
                        category = 'moderate'
                
                # 4. On surface and far away (> 15 A) -> FLEXIBLE
                else:
                    category = 'flexible'
                
                classification[res_num] = {
                    'residue_name': res_name,
                    'category': category,
                    'distance_to_active_site': distances.get(res_num, None),
                    'accessibility': accessibility.get(res_num, (None, None))
                }
        
        print(f"  {len(classification)} amino acids classified")
        return classification
    
    def generate_directed_evolution_mask(self, classification):
        """
        Generate mask for directed evolution
        """
        mask = {
            'fixed': [],      # Must not be changed
            'moderate': [],   # Can be changed carefully
            'flexible': []    # Can be changed freely
        }
        
        for res_num, info in classification.items():
            category = info['category']
            mask[category].append(res_num)
        
        # Sort
        for key in mask:
            mask[key] = sorted(mask[key])
        
        return mask
    
    def print_summary(self, classification, mask):
        """
        Print analysis summary
        """
        print("\n" + "="*80)
        print("CRITICAL RESIDUE ANALYSIS SUMMARY")
        print("="*80)
        
        total = len(classification)
        fixed_count = len(mask['fixed'])
        moderate_count = len(mask['moderate'])
        flexible_count = len(mask['flexible'])
        
        print(f"\n Total Amino Acids: {total}")
        print(f" Fixed: {fixed_count} ({fixed_count/total*100:.1f}%)")
        print(f" Moderate: {moderate_count} ({moderate_count/total*100:.1f}%)")
        print(f" Flexible: {flexible_count} ({flexible_count/total*100:.1f}%)")

        print(f"\n Catalytic Triad: {self.catalytic_triad}")
        print(f" Oxyanion Hole: {self.oxyanion_hole}")
        print(f" Substrate Binding: {self.substrate_binding}")

def generate_contribution_graph(mask, report_folder):
    """
    GitHub contribution graph style visualization
    """
    print("\n-> Generating contribution graph...")
    
    n_positions = 300
    weeks = 52
    positions_per_week = n_positions // weeks
    
    # Mark all positions
    color_map = {
        'fixed': '#d73a49',      # Red
        'moderate': '#ffd33d',   # Yellow
        'flexible': '#2ea44f',   # Green
        'empty': '#ebedf0'       # Gray
    }
    
    # Assign color to each position
    position_colors = {}
    for i in range(1, n_positions + 1):
        if i in mask['fixed']:
            position_colors[i] = color_map['fixed']
        elif i in mask['moderate']:
            position_colors[i] = color_map['moderate']
        elif i in mask['flexible']:
            position_colors[i] = color_map['flexible']
        else:
            position_colors[i] = color_map['empty']
    
    # Create grid
    fig, ax = plt.subplots(figsize=(16, 4))
    
    n_rows = 7
    n_cols = n_positions // n_rows + 1
    
    for pos in range(1, n_positions + 1):
        row = (pos - 1) % n_rows
        col = (pos - 1) // n_rows
        
        color = position_colors.get(pos, color_map['empty'])
        rect = plt.Rectangle((col, n_rows - row - 1), 1, 1, 
                            facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
    
    # Axis settings
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Week labels (every 4 weeks)
    for i in range(0, n_cols, 4):
        ax.text(i, -0.5, f'W{i}', ha='left', va='top', fontsize=7, color='gray')
    
    # Row labels
    y_labels = ['Mon', '', 'Wed', '', 'Fri', '', 'Sun']
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Title
    ax.set_title('Enzyme Residue Classification Map', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Amino Acid Position', fontsize=10)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['fixed'], edgecolor='none', label=f'Fixed ({len(mask["fixed"])})'),
        Patch(facecolor=color_map['moderate'], edgecolor='none', label=f'Moderate ({len(mask["moderate"])})'),
        Patch(facecolor=color_map['flexible'], edgecolor='none', label=f'Flexible ({len(mask["flexible"])})'),
        Patch(facecolor=color_map['empty'], edgecolor='none', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(1.15, 1), fontsize=9)
    
    # Statistics box
    total = len(mask['fixed']) + len(mask['flexible']) + len(mask['moderate'])
    stats_text = f"Total: {total} residues\n"
    stats_text += f"Fixed: {len(mask['fixed'])/total*100:.1f}%\n"
    stats_text += f"Flexible: {len(mask['flexible'])/total*100:.1f}%"
    
    ax.text(1.02, 0.3, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    graph_path = os.path.join(report_folder, 'residue_contribution_graph.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Contribution graph saved: {graph_path}")

# ============================================================================
# SECTION 3: MODEL LOADING (PyCaret)
# ============================================================================

# --- USER SETTINGS ---
MODEL_NAME = "topt_automl_model"
FULL_MODEL_PATH = os.path.join("report", MODEL_NAME)

print("\n" + "="*80)
print("STEP 3: LOADING MACHINE LEARNING MODEL")
print("="*80)

print(f"Loading model: {FULL_MODEL_PATH} ... ", end="")
try:
    pycaret_model = load_model(FULL_MODEL_PATH, verbose=False)
    print("Success.")
except Exception as e:
    print(f"\n ERROR: Could not load model! {e}")
    print(f"   Path: {FULL_MODEL_PATH}")
    exit()

# ============================================================================
# SECTION 4: ADVANCED DIRECTED EVOLUTION FUNCTIONS
# ============================================================================

# Initial amino acid sequence (Wild-type)
initial_sequence = "ANPYERGPNPTDALLEARSGPFSVSEENVSRLSASGFGGGTIYYPRENNTYGAVAISPGYTGTEASIAWLGERIASHGFVVITIDTITTLDQPDSRAEQLNAALNHMINRASSTVRSRIDSSRLAVMGHSMGGGGSLRLASQRPDLKAAIPLTPWHLNKNWSSVTVPTLIIGADLDTIAPVATHAKPFYNSLPSSISKAYLELDGATHFAPNIPNKIIGKYSVAWLKRFVDNDTRYTQFLCPGPRDGLFGEVEEYRSTCPF"

def calculate_adaptive_mutation_rate(current_topt, target_topt, base_rate=0.05, max_rate=0.15):
    """
    Calculate adaptive mutation rate based on distance to target

    Far from target -> High mutation (exploration)
    Close to target -> Low mutation (fine-tuning)
    """
    distance = abs(current_topt - target_topt)
    
    # 0-5 C range: linearly decreasing mutation
    if distance <= 5.0:
        rate = base_rate + (max_rate - base_rate) * (distance / 5.0)
    else:
        rate = max_rate
    
    return min(rate, max_rate)

def mutate_sequence_multiscale(sequence, fixed_positions, flexible_positions, 
                                mutation_rate=0.05, aggressive=False):
    """
    Multi-scale mutation strategy

    Args:
        sequence: Amino acid sequence
        fixed_positions: Positions to keep fixed (1-indexed)
        flexible_positions: Positions that can be changed (1-indexed)
        mutation_rate: Mutation rate
        aggressive: If True, make more changes

    Returns:
        Mutant amino acid sequence
    """
    sequence = list(sequence)
    
    # In aggressive mode, change more positions
    if aggressive:
        mutation_rate = min(mutation_rate * 2, 0.3)
    
    # Only change flexible positions
    mutated_positions = []
    for pos in flexible_positions:
        if np.random.rand() < mutation_rate:
            idx = pos - 1
            
            if 0 <= idx < len(sequence):
                current_aa = sequence[idx]
                # Select a different amino acid
                possible_aa = [aa for aa in elib.AMINO_ACIDS if aa != current_aa]
                new_aa = np.random.choice(list(possible_aa))
                sequence[idx] = new_aa
                mutated_positions.append(pos)
    
    return ''.join(sequence), len(mutated_positions)

def generate_diverse_population(base_sequence, fixed_positions, flexible_positions, 
                                population_size=10, mutation_rate=0.1):
    """
    Generate new individuals for diversity
    """
    diverse_population = []
    
    for _ in range(population_size):
        mutant, _ = mutate_sequence_multiscale(
            base_sequence,
            fixed_positions,
            flexible_positions,
            mutation_rate=mutation_rate,
            aggressive=True
        )
        diverse_population.append(mutant)
    
    return diverse_population

# ============================================================================
# SECTION 5: ADVANCED DIRECTED EVOLUTION FUNCTION
# ============================================================================

def directed_evolution_improved(initial_sequence, critical_mask, target_topt=24.0,
                               current_topt=None,
                               n_iterations=150, population_size=300,
                               base_mutation_rate=0.05, convergence_threshold=0.01,
                               diversity_injection_interval=15):
    """
    Improved directed evolution algorithm

    IMPROVEMENTS:
    1. Adaptive mutation rate
    2. Periodic diversity injection
    3. Multi-scale mutation
    4. Advanced convergence detection

    Args:
        initial_sequence: Initial amino acid sequence
        critical_mask: {'fixed': [], 'moderate': [], 'flexible': []}
        target_topt: Target temperature (C)
        n_iterations: Maximum number of iterations
        population_size: Number of mutants to generate per iteration
        base_mutation_rate: Base mutation rate
        convergence_threshold: Convergence threshold (C)
        diversity_injection_interval: How many iterations between diversity injections

    Returns:
        best_sequence, best_topt, topt_history
    """
    print("\n" + "="*80)
    print("STEP 4: ADVANCED DIRECTED EVOLUTION")
    print("="*80)
    
    print(f"\n Parameters:")
    print(f"   Target Topt: {target_topt}C")
    print(f"   Maximum iterations: {n_iterations}")
    print(f"   Population size: {population_size}")
    print(f"   Base mutation rate: {base_mutation_rate}")
    print(f"   Diversity injection: Every {diversity_injection_interval} iterations")
    
    # Fitness function
    def fitness(topt_value):
        return 1.0 / (abs(topt_value - target_topt) + 0.1)
    
    # Initial evaluation
    print("\n-> Evaluating initial sequence...")
    start_features = elib.calculate_all_features(initial_sequence)
    pred_col = 'prediction_label'

    if current_topt is not None:
        start_topt = current_topt
        print(f"Initial Topt (from dataset): {start_topt:.2f}C")
    else:
        start_features_df = pd.DataFrame([start_features])
        start_pred = predict_model(pycaret_model, data=start_features_df, verbose=False)
        start_topt = start_pred[pred_col].values[0]
        print(f"Initial Topt (model prediction): {start_topt:.2f}C")

    print(f"Deviation from target: {abs(start_topt - target_topt):.2f}C")
    
    # Track best values
    best_sequence = initial_sequence
    best_topt = start_topt
    best_fitness = fitness(start_topt)
    best_features = start_features
    
    topt_history = [start_topt]
    mutation_rate_history = []
    no_improvement_counter = 0
    
    start_time = time.time()
    
    print("\n Evolution starting...")
    print("="*80)
    
    for iteration in range(n_iterations):
        # Early termination check
        current_deviation = abs(best_topt - target_topt)
        if current_deviation < 0.5:
            print(f"\n TARGET TOPT VALUE REACHED! (Deviation: {current_deviation:.2f}C)")
            print(f"   Stopping at iteration {iteration + 1}/{n_iterations}.")
            break
        
        iter_start_time = time.time()
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        
        # Calculate adaptive mutation rate
        current_mutation_rate = calculate_adaptive_mutation_rate(
            best_topt, target_topt, base_mutation_rate, max_rate=0.15
        )
        mutation_rate_history.append(current_mutation_rate)
        
        print(f"  Adaptive mutation rate: {current_mutation_rate:.3f}")
        
        # Diversity injection (when stuck)
        inject_diversity = False
        if iteration > 0 and iteration % diversity_injection_interval == 0:
            inject_diversity = True
            print(f"  Diversity injection active!")
        
        # Convergence detection (no improvement counter)
        if no_improvement_counter > 20:
            inject_diversity = True
            print(f"    No improvement for 20 iterations, injecting diversity!")
            no_improvement_counter = 0
        
        # Generate mutants
        mutants = []
        
        # Normal mutations
        normal_mutants = int(population_size * 0.8) if inject_diversity else population_size
        for _ in range(normal_mutants):
            mutant, _ = mutate_sequence_multiscale(
                best_sequence,
                critical_mask['fixed'],
                critical_mask['flexible'],
                mutation_rate=current_mutation_rate,
                aggressive=False
            )
            mutants.append(mutant)
        
        # If diversity injection is active
        if inject_diversity:
            diverse_mutants = generate_diverse_population(
                best_sequence,
                critical_mask['fixed'],
                critical_mask['flexible'],
                population_size=int(population_size * 0.2),
                mutation_rate=0.15
            )
            mutants.extend(diverse_mutants)
        
        # Evaluate mutants - BATCH processing
        mutant_features_list = []
        valid_mutants = []
        
        for mutant in mutants:
            mutant_features = elib.calculate_all_features(mutant)
            
            if mutant_features is not None and mutant_features.get('length', 0) > 0:
                mutant_features_list.append(mutant_features)
                valid_mutants.append(mutant)
        
        if not mutant_features_list:
            print("    No valid mutant found, skipping...")
            continue
        
        # Batch prediction with PyCaret
        mutant_features_df = pd.DataFrame(mutant_features_list)
        mutant_preds = predict_model(pycaret_model, data=mutant_features_df, verbose=False)
        mutant_topt_values_arr = mutant_preds[pred_col].values
        
        # Find the best mutant
        mutant_topt_values = []
        for i, (mutant, mutant_topt) in enumerate(zip(valid_mutants, mutant_topt_values_arr)):
            mutant_fitness_val = fitness(mutant_topt)
            mutant_topt_values.append((mutant, mutant_topt, mutant_fitness_val, mutant_features_list[i]))
        
        # Select the best mutant
        mutant_topt_values.sort(key=lambda x: x[2], reverse=True)
        
        if mutant_topt_values:
            best_mutant, best_mutant_topt, best_mutant_fitness, best_mutant_features = mutant_topt_values[0]
            
            if best_mutant_fitness > best_fitness:
                improvement = abs(best_mutant_topt - best_topt)
                best_sequence = best_mutant
                best_topt = best_mutant_topt
                best_fitness = best_mutant_fitness
                best_features = best_mutant_features
                no_improvement_counter = 0
                
                print(f"  Improvement! New Topt: {best_topt:.2f}C (Change: {improvement:.2f}C)")
                
                # Convergence check
                if improvement < convergence_threshold and abs(best_topt - target_topt) < 1.0:
                    print(f"    Convergence detected, {abs(best_topt - target_topt):.2f}C away from target")
                    # Continue anyway, may escape local optimum
            else:
                no_improvement_counter += 1
                print(f"    No improvement ({no_improvement_counter}). Best Topt: {best_topt:.2f}C")
        
        topt_history.append(best_topt)
        
        iter_time = time.time() - iter_start_time
        print(f"   Iteration time: {iter_time:.2f} seconds")
        print(f"   Deviation from target: {abs(best_topt - target_topt):.2f}C")
        print(f"   Number of valid mutants: {len(valid_mutants)}")
    
    total_time = time.time() - start_time
    print(f"\nEvolution process completed! Total time: {total_time:.2f} seconds")
    
    return best_sequence, best_topt, topt_history, mutation_rate_history

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def parse_int_list(value):
    """Parse a comma-separated string of integers into a list."""
    if not value or value.strip() == '':
        return []
    return [int(x.strip()) for x in value.split(',')]

def md_to_html(md_path, html_path):
    """Converts a markdown report to a standalone HTML file with embedded images."""
    report_dir = os.path.dirname(os.path.abspath(md_path))

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # --- Embed images as base64 ---
    def replace_image(match):
        alt_text = match.group(1)
        img_file = match.group(2)
        img_path = os.path.join(report_dir, img_file)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as img_f:
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
            ext = os.path.splitext(img_file)[1].lstrip('.').lower()
            mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            return f'<img src="data:{mime};base64,{b64}" alt="{alt_text}" style="max-width:100%;">'
        return match.group(0)

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, md_text)

    # --- Simple markdown to HTML conversion ---
    html_lines = []
    in_table = False
    in_list = False
    lines = md_text.split('\n')

    for line in lines:
        stripped = line.strip()

        # Headings
        if stripped.startswith('### '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h3>{stripped[4:]}</h3>')
            continue
        if stripped.startswith('## '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h2>{stripped[3:]}</h2>')
            continue
        if stripped.startswith('# '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h1>{stripped[2:]}</h1>')
            continue

        # Already-converted <img> tags
        if stripped.startswith('<img '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(stripped)
            continue

        # Table rows
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if all(re.match(r'^-+$', c) for c in cells):
                continue
            if not in_table:
                in_table = True
                html_lines.append('<table border="1" cellpadding="6" cellspacing="0" '
                                  'style="border-collapse:collapse; margin:10px 0;">')
                html_lines.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
            else:
                html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            continue
        else:
            if in_table:
                html_lines.append('</table>')
                in_table = False

        # Ordered list items (1. item)
        list_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if list_match:
            if not in_list:
                in_list = True
                html_lines.append('<ol>')
            html_lines.append(f'<li>{list_match.group(2)}</li>')
            continue
        else:
            if in_list:
                html_lines.append('</ol>')
                in_list = False

        # Bold: **text**
        stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)

        # Bullet list item: - text
        if stripped.startswith('- '):
            html_lines.append(f'<p style="margin:2px 0 2px 20px;">{stripped[2:]}</p>')
            continue

        # Code block line (4 spaces or backtick)
        if line.startswith('    ') or stripped.startswith('`'):
            html_lines.append(f'<code style="font-family:Courier New,monospace; '
                              f'background:#f4f4f4; padding:1px 4px;">{stripped.strip("`")}</code><br>')
            continue

        # Empty line
        if not stripped:
            continue

        # Regular paragraph
        html_lines.append(f'<p>{stripped}</p>')

    if in_table:
        html_lines.append('</table>')
    if in_list:
        html_lines.append('</ol>')

    body = '\n'.join(html_lines)

    html_doc = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Directed Evolution Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }}
  h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
  h3 {{ color: #7f8c8d; margin-top: 20px; }}
  table {{ font-size: 13px; }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f2f2; }}
  img {{ display: block; margin: 10px 0 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
  .seq {{ font-family: Courier New, monospace; word-break: break-all; background: #f9f9f9;
          padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    print(f"HTML report saved: {html_path}")


def main():
    """
    Main program flow
    """
    parser = argparse.ArgumentParser(description='Directed Evolution')
    parser.add_argument('--target-topt', type=float, default=35.0,
                        help='Target Topt temperature (default: 35.0)')
    parser.add_argument('--current-topt', type=float, default=None,
                        help='Current Topt from dataset (used as initial Topt instead of model prediction)')
    parser.add_argument('--dataset-csv', type=str, default='',
                        help='Dataset CSV file to lookup sequence by UniProt ID')
    parser.add_argument('--uniprot-id', type=str, default='E5BBQ3',
                        help='UniProt accession ID (default: E5BBQ3)')
    parser.add_argument('--catalytic-triad', type=str, default='130,176,208',
                        help='Catalytic triad positions, comma-separated (default: 130,176,208)')
    parser.add_argument('--oxyanion-hole', type=str, default='60,131,132',
                        help='Oxyanion hole positions, comma-separated (default: 60,131,132)')
    parser.add_argument('--substrate-binding', type=str, default='',
                        help='Substrate binding positions, comma-separated (default: empty)')
    parser.add_argument('--iterations', type=int, default=150,
                        help='Max iterations (default: 150)')
    parser.add_argument('--population', type=int, default=300,
                        help='Population size (default: 300)')
    parser.add_argument('--mutation-rate', type=float, default=0.05,
                        help='Base mutation rate (default: 0.05)')
    parser.add_argument('--sequence', type=str, default='',
                        help='Initial sequence (overrides CSV lookup if provided)')
    args = parser.parse_args()

    target_topt = args.target_topt
    catalytic_triad = parse_int_list(args.catalytic_triad)
    oxyanion_hole = parse_int_list(args.oxyanion_hole)
    substrate_binding = parse_int_list(args.substrate_binding)

    print("\n" + "="*80)
    print("ENZYME COLD/HEAT ADAPTATION PROJECT - IMPROVED VERSION")
    print("Gokce Ceyda Bilgin - ISEF 2026")
    print("="*80)

    # Resolve initial sequence: --sequence > CSV lookup > hardcoded fallback
    seq = None
    if args.sequence.strip():
        seq = args.sequence.strip()
        print(f"\nUsing provided sequence (length: {len(seq)} amino acids)")
    elif args.dataset_csv and os.path.exists(args.dataset_csv):
        print(f"\nLooking up UniProt ID '{args.uniprot_id}' in {args.dataset_csv}...")
        try:
            for sep in [';', ',']:
                try:
                    df_lookup = pd.read_csv(args.dataset_csv, sep=sep)
                    if 'uniprot_id' in df_lookup.columns:
                        break
                except:
                    continue
            if 'uniprot_id' in df_lookup.columns and 'sequence' in df_lookup.columns:
                match = df_lookup[df_lookup['uniprot_id'] == args.uniprot_id]
                if not match.empty:
                    seq = str(match.iloc[0]['sequence']).strip()
                    print(f"   Sequence found! Length: {len(seq)} amino acids")
                else:
                    print(f"   ERROR: UniProt ID '{args.uniprot_id}' not found in dataset!")
                    print(f"   Available UniProt IDs (first 10): {list(df_lookup['uniprot_id'].head(10))}")
                    sys.exit(1)
            else:
                print(f"   ERROR: CSV does not contain 'uniprot_id' or 'sequence' columns!")
                sys.exit(1)
        except Exception as e:
            print(f"   ERROR: Could not read dataset CSV: {e}")
            sys.exit(1)
    else:
        # Fallback to hardcoded default sequence
        seq = initial_sequence
        print(f"\nNo dataset CSV provided, using default sequence (length: {len(seq)})")

    # Create report folder
    report_folder = create_report_dir()

    # 1. Resolve UniProt ID and download PDB file
    pdb_file, pdb_id = download_pdb_structure(uniprot_id=args.uniprot_id)
    if not pdb_file:
        print("\n PDB file could not be loaded. Check UniProt ID.")
        print("Terminating program...")
        return

    # 2. Analyze critical residues
    analyzer = CriticalResidueAnalyzer(pdb_file,
                                       catalytic_triad=catalytic_triad,
                                       oxyanion_hole=oxyanion_hole,
                                       substrate_binding=substrate_binding)
    classification = analyzer.classify_residues()
    mask = analyzer.generate_directed_evolution_mask(classification)

    # Show summary
    analyzer.print_summary(classification, mask)

    # Save mask
    mask_path = os.path.join(report_folder, 'critical_residues_mask.json')
    with open(mask_path, 'w', encoding='utf-8') as f:
        json.dump(mask, f, indent=2, ensure_ascii=False)
    print(f"\nCritical residue mask saved: {mask_path}")

    # GitHub Contribution Graph visualization
    generate_contribution_graph(mask, report_folder)

    # 3. Improved directed evolution
    best_sequence, best_topt, topt_history, mutation_rate_history = directed_evolution_improved(
        initial_sequence=seq,
        critical_mask=mask,
        target_topt=target_topt,
        current_topt=args.current_topt,
        n_iterations=args.iterations,
        population_size=args.population,
        base_mutation_rate=args.mutation_rate,
        convergence_threshold=0.01,
        diversity_injection_interval=15
    )
    
    # 4. Visualize results
    if topt_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Topt change plot
        ax1.plot(range(len(topt_history)), topt_history, marker='o', linestyle='-',
                color='b', linewidth=2, markersize=4)
        ax1.axhline(y=target_topt, color='r', linestyle='--', linewidth=2,
                   label=f"Target Topt ({target_topt:.1f}C)")
        ax1.set_title("Improved Directed Evolution - Topt Change",
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Topt Value (C)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Mutation rate plot
        if mutation_rate_history:
            ax2.plot(range(len(mutation_rate_history)), mutation_rate_history,
                    marker='s', linestyle='-', color='g', linewidth=2, markersize=3)
            ax2.set_title("Adaptive Mutation Rate", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Iteration", fontsize=12)
            ax2.set_ylabel("Mutation Rate", fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        graph_path = os.path.join(report_folder, 'evolution_plot_improved.png')
        plt.savefig(graph_path, dpi=300)
        plt.close()
        print(f"\nPlot saved: {graph_path}")

    # 5. Comparison plot (original vs improved)
    # Load original plot
    try:
        original_graph = plt.imread('/mnt/user-data/uploads/evrim_grafik_biopython.png')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original
        ax1.imshow(original_graph)
        ax1.axis('off')
        ax1.set_title("Original Algorithm", fontsize=14, fontweight='bold')

        # Improved
        ax2.plot(range(len(topt_history)), topt_history, marker='o', linestyle='-',
                color='b', linewidth=2, markersize=4)
        ax2.axhline(y=target_topt, color='r', linestyle='--', linewidth=2,
                   label=f"Target Topt ({target_topt:.1f}C)")
        ax2.set_title("Improved Algorithm", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("Topt Value (C)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        comparison_path = os.path.join(report_folder, 'comparison_graph.png')
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        print(f"Comparison plot saved: {comparison_path}")
    except:
        print("  Original plot could not be loaded, comparison skipped")
    
    # 6. Save results
    if best_sequence and best_topt:
        print("\n" + "="*80)
        print("FINAL RESULTS (IMPROVED BIOPYTHON + PYCARET)")
        print("="*80)
        print(f"\n Initial Topt: {topt_history[0]:.2f}C")
        print(f" Final Topt: {best_topt:.2f}C")
        print(f" Total Improvement: {abs(best_topt - topt_history[0]):.2f}C")
        print(f" Deviation from Target: {abs(best_topt - target_topt):.2f}C")

        # Calculate success rate
        total_distance = abs(topt_history[0] - target_topt)
        achieved_distance = abs(best_topt - target_topt)
        success_rate = (1 - achieved_distance / total_distance) * 100
        print(f" Success Rate: {success_rate:.1f}%")

        print("\n Best Amino Acid Sequence:")
        for i in range(0, len(best_sequence), 80):
            print(best_sequence[i:i+80])

        # Save to file
        result_path = os.path.join(report_folder, 'best_enzyme_improved.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("ENZYME COLD/HEAT ADAPTATION PROJECT - IMPROVED VERSION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Topt Value: {best_topt:.2f}C\n")
            f.write(f"Initial: {topt_history[0]:.2f}C\n")
            f.write(f"Improvement: {abs(best_topt - topt_history[0]):.2f}C\n")
            f.write(f"Deviation from Target: {abs(best_topt - target_topt):.2f}C\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n\n")

            f.write("IMPROVEMENTS:\n")
            f.write("1. Adaptive mutation rate (based on distance to target)\n")
            f.write("2. Diversity preservation (every 15 iterations)\n")
            f.write("3. Multi-scale mutation strategy\n")
            f.write("4. Improved convergence detection\n\n")

            f.write("Amino Acid Sequence:\n")
            for i in range(0, len(best_sequence), 80):
                f.write(f"{best_sequence[i:i+80]}\n")

            f.write("\n\nCritical Position Statistics:\n")
            f.write(f"Fixed: {len(mask['fixed'])} amino acids\n")
            f.write(f"Modified: {len(mask['flexible'])} amino acids\n")
            f.write(f"Carefully modified: {len(mask['moderate'])} amino acids\n")

        print(f"\nResults saved: {result_path}")

        # Save sequence-only file (for Step 4)
        seq_only_path = os.path.join(report_folder, 'best_sequence.txt')
        with open(seq_only_path, 'w', encoding='utf-8') as f:
            f.write(best_sequence)
        print(f"Sequence-only file saved: {seq_only_path}")

        # Analyze changes
        changes = []
        for i, (orig, new) in enumerate(zip(seq, best_sequence)):
            if orig != new:
                changes.append((i+1, orig, new))

        print(f"\n Total {len(changes)} positions changed")
        if changes:
            print("\nFirst 20 changes:")
            for pos, orig, new in changes[:20]:
                in_fixed = "IN FIXED REGION!" if pos in mask['fixed'] else ""
                print(f"  Position {pos}: {orig} -> {new} {in_fixed}")

        # ============================================================
        # 7. Generate Markdown + HTML Report
        # ============================================================
        print("\n" + "="*80)
        print("GENERATING HTML REPORT")
        print("="*80)

        md_path = os.path.join(report_folder, 'Directed_Evolution_Report.md')
        md_lines = []

        md_lines.append("# Directed Evolution Report")
        md_lines.append("")
        md_lines.append("**Enzyme Cold/Heat Adaptation Project - Improved Version**")
        md_lines.append("")
        md_lines.append("Gokce Ceyda Bilgin - ISEF 2026")
        md_lines.append("")

        # --- Parameters ---
        md_lines.append("## Parameters")
        md_lines.append("")
        md_lines.append(f"| Parameter | Value |")
        md_lines.append(f"| --- | --- |")
        md_lines.append(f"| UniProt ID | {args.uniprot_id} |")
        md_lines.append(f"| Target Topt | {target_topt}C |")
        md_lines.append(f"| Initial Topt | {topt_history[0]:.2f}C |")
        md_lines.append(f"| Deviation from target | {abs(topt_history[0] - target_topt):.2f}C |")
        md_lines.append(f"| Maximum iterations | {args.iterations} |")
        md_lines.append(f"| Population size | {args.population} |")
        md_lines.append(f"| Base mutation rate | {args.mutation_rate} |")
        md_lines.append(f"| Diversity injection | Every 15 iterations |")
        md_lines.append("")

        # --- Final Results ---
        md_lines.append("## Final Results")
        md_lines.append("")
        md_lines.append(f"| Metric | Value |")
        md_lines.append(f"| --- | --- |")
        md_lines.append(f"| Initial Topt | {topt_history[0]:.2f}C |")
        md_lines.append(f"| Final Topt | {best_topt:.2f}C |")
        md_lines.append(f"| Total Improvement | {abs(best_topt - topt_history[0]):.2f}C |")
        md_lines.append(f"| Deviation from Target | {abs(best_topt - target_topt):.2f}C |")
        md_lines.append(f"| Success Rate | {success_rate:.1f}% |")
        md_lines.append(f"| Positions Changed | {len(changes)} |")
        md_lines.append("")

        # --- Critical Residue Statistics ---
        md_lines.append("## Critical Residue Statistics")
        md_lines.append("")
        total_res = len(mask['fixed']) + len(mask['moderate']) + len(mask['flexible'])
        md_lines.append(f"| Category | Count | Percentage |")
        md_lines.append(f"| --- | --- | --- |")
        md_lines.append(f"| Fixed | {len(mask['fixed'])} | {len(mask['fixed'])/total_res*100:.1f}% |")
        md_lines.append(f"| Moderate | {len(mask['moderate'])} | {len(mask['moderate'])/total_res*100:.1f}% |")
        md_lines.append(f"| Flexible | {len(mask['flexible'])} | {len(mask['flexible'])/total_res*100:.1f}% |")
        md_lines.append(f"| **Total** | **{total_res}** | **100%** |")
        md_lines.append("")

        # --- Mutation Changes ---
        if changes:
            md_lines.append("## Mutation Changes")
            md_lines.append("")
            md_lines.append(f"| Position | Original | New | Region |")
            md_lines.append(f"| --- | --- | --- | --- |")
            for pos, orig, new in changes:
                region = "Fixed" if pos in mask['fixed'] else ("Moderate" if pos in mask['moderate'] else "Flexible")
                md_lines.append(f"| {pos} | {orig} | {new} | {region} |")
            md_lines.append("")

        # --- Best Sequence ---
        md_lines.append("## Best Enzyme Sequence")
        md_lines.append("")
        md_lines.append(f"**Length:** {len(best_sequence)} amino acids")
        md_lines.append("")
        # Wrap sequence 80 chars per line in code block style
        md_lines.append("```")
        for i in range(0, len(best_sequence), 80):
            md_lines.append(best_sequence[i:i+80])
        md_lines.append("```")
        md_lines.append("")

        # --- Plots ---
        md_lines.append("## Evolution Plot")
        md_lines.append("")
        md_lines.append("![Evolution Plot](evolution_plot_improved.png)")
        md_lines.append("")

        md_lines.append("## Residue Contribution Graph")
        md_lines.append("")
        md_lines.append("![Residue Contribution Graph](residue_contribution_graph.png)")
        md_lines.append("")

        # Save markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        print(f"Markdown report saved: {md_path}")

        # Convert to HTML
        html_path = os.path.join(report_folder, 'Directed_Evolution_Report.html')
        md_to_html(md_path, html_path)

    print("\n" + "="*80)
    print("PROGRAM COMPLETED!")
    print("="*80)

def cleanup_pycache():
    """Delete __pycache__ folder"""
    if os.path.exists('__pycache__'):
        try:
            shutil.rmtree('__pycache__')
            print("__pycache__ folder deleted.")
        except Exception as e:
            print(f"  Could not delete __pycache__: {e}")

if __name__ == "__main__":
    main()
    cleanup_pycache()