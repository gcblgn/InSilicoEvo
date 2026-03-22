#!/usr/bin/env python3
"""
ESMFold Protein Structure Prediction Tool
==========================================

A local tool for protein structure prediction using ESMFold API.
No GPU required - all computation happens on Meta's servers.

Requirements:
    pip install requests biopython matplotlib plotly pandas seaborn

Usage:
    python esmfold_structure_prediction.py

Author: Computational Enzyme Engineering Project
For: ISEF 2026 - Cold-Adapted Enzyme Design
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import webbrowser
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# CORE PREDICTION FUNCTIONS
# =============================================================================

def predict_structure_esmfold(sequence: str, name: str = "protein") -> Optional[Dict]:
    """
    Predict protein structure using ESMFold API.
    
    This function calls Meta's ESMFold API - no local GPU needed.
    The computation happens on Meta's servers.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence (can contain whitespace/newlines)
    name : str
        Name/label for the protein
    
    Returns
    -------
    dict or None
        Dictionary containing:
        - pdb_string: PDB format structure
        - plddt_scores: Per-residue confidence scores
        - mean_plddt: Average confidence
        - quality: Quality assessment string
        Returns None if prediction fails
    
    Example
    -------
    >>> result = predict_structure_esmfold("MKTAYIAKQRQISFVK", "test_protein")
    >>> print(result["mean_plddt"])
    85.3
    """
    # Clean sequence
    sequence = ''.join(sequence.split()).upper()
    
    if not sequence:
        print("Error: Empty sequence provided")
        return None
    
    # Validate sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        print(f"Warning: Invalid characters found: {invalid_chars}")
        print("Removing invalid characters...")
        sequence = ''.join(c for c in sequence if c in valid_aa)
    
    print(f"Predicting structure for {name} ({len(sequence)} residues)...")
    start_time = time.time()
    
    # ESMFold API endpoint
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    try:
        response = requests.post(
            url, 
            data=sequence, 
            timeout=300,
            headers={"Content-Type": "text/plain"}
        )
        response.raise_for_status()
        pdb_string = response.text
        
    except requests.exceptions.Timeout:
        print("Error: API request timed out. Try a shorter sequence.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed - {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"Structure predicted in {elapsed:.1f} seconds")
    
    # Extract pLDDT scores from B-factor column (with debug info)
    # Also get corrected PDB string with 0-100 scale B-factors
    plddt_scores, corrected_pdb = extract_plddt_from_pdb(pdb_string, debug=True)
    
    if len(plddt_scores) == 0:
        print("Error: Could not extract pLDDT scores from PDB")
        return None
    
    # Use corrected PDB for visualization (B-factors in 0-100 scale)
    pdb_string = corrected_pdb
    
    # Calculate statistics
    mean_plddt = np.mean(plddt_scores)
    min_plddt = np.min(plddt_scores)
    max_plddt = np.max(plddt_scores)
    std_plddt = np.std(plddt_scores)
    
    # Quality assessment
    if mean_plddt >= 90:
        quality = "Excellent"
        quality_color = "#2ecc71"
    elif mean_plddt >= 70:
        quality = "Good"
        quality_color = "#3498db"
    elif mean_plddt >= 50:
        quality = "Moderate"
        quality_color = "#f39c12"
    else:
        quality = "Poor"
        quality_color = "#e74c3c"
    
    print(f"Mean pLDDT: {mean_plddt:.1f} ({quality})")
    
    return {
        "name": name,
        "sequence": sequence,
        "length": len(sequence),
        "pdb_string": pdb_string,
        "plddt_scores": plddt_scores,
        "mean_plddt": mean_plddt,
        "min_plddt": min_plddt,
        "max_plddt": max_plddt,
        "std_plddt": std_plddt,
        "quality": quality,
        "quality_color": quality_color,
        "prediction_time": elapsed
    }


def extract_plddt_from_pdb(pdb_string: str, debug: bool = False) -> tuple:
    """
    Extract pLDDT scores from PDB B-factor column.
    
    ESMFold stores pLDDT confidence scores in the B-factor field
    of the PDB file. This function extracts one score per residue
    (using CA atoms).
    
    Note: ESMFold API sometimes returns values in 0-1 scale instead
    of 0-100. This function automatically detects and converts.
    
    Parameters
    ----------
    pdb_string : str
        PDB format string from ESMFold
    debug : bool
        If True, print diagnostic information
    
    Returns
    -------
    tuple
        (plddt_array, corrected_pdb_string)
        - plddt_array: numpy array of pLDDT scores (0-100 scale)
        - corrected_pdb_string: PDB string with B-factors in 0-100 scale
    """
    plddt_scores = []
    seen_residues = set()
    
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            try:
                # Get residue number to avoid duplicates
                res_num = int(line[22:26].strip())
                if res_num not in seen_residues:
                    seen_residues.add(res_num)
                    # B-factor is in columns 61-66
                    b_factor = float(line[60:66].strip())
                    plddt_scores.append(b_factor)
            except (ValueError, IndexError):
                continue
    
    plddt_array = np.array(plddt_scores)
    
    if len(plddt_array) == 0:
        return plddt_array, pdb_string
    
    # Debug output
    if debug:
        print(f"  DEBUG: Extracted {len(plddt_array)} pLDDT values")
        print(f"  DEBUG: Raw range: {plddt_array.min():.4f} - {plddt_array.max():.4f}")
    
    # Auto-detect scale: if max <= 1, values are in 0-1 scale
    # ESMFold API sometimes returns 0-1 instead of 0-100
    needs_conversion = plddt_array.max() <= 1.0
    
    if needs_conversion:
        if debug:
            print("  DEBUG: Detected 0-1 scale, converting to 0-100")
        plddt_array = plddt_array * 100
        
        # Also fix the PDB string so 3D visualization works correctly
        corrected_lines = []
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    # B-factor is in columns 61-66
                    old_bfactor = float(line[60:66].strip())
                    new_bfactor = old_bfactor * 100
                    # Format B-factor with proper width (6 chars, 2 decimals)
                    new_bfactor_str = f"{new_bfactor:6.2f}"
                    # Replace in line
                    line = line[:60] + new_bfactor_str + line[66:]
                except (ValueError, IndexError):
                    pass
            corrected_lines.append(line)
        corrected_pdb = '\n'.join(corrected_lines)
    else:
        corrected_pdb = pdb_string
    
    if debug:
        print(f"  DEBUG: Final range: {plddt_array.min():.1f} - {plddt_array.max():.1f}")
    
    return plddt_array, corrected_pdb


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_3d_html_viewer(result: Dict, output_path: str = None, 
                          auto_open: bool = True) -> str:
    """
    Create interactive 3D visualization as HTML file.
    
    Since we're running locally (not in Jupyter), this creates an HTML
    file that can be opened in any web browser.
    
    Parameters
    ----------
    result : dict
        Output from predict_structure_esmfold
    output_path : str, optional
        Path for HTML file. Default: {name}_3d_structure.html
    auto_open : bool
        Whether to automatically open in browser
    
    Returns
    -------
    str
        Path to created HTML file
    """
    if output_path is None:
        output_path = f"{result['name']}_3d_structure.html"
    
    # Escape PDB string for JavaScript
    pdb_escaped = result['pdb_string'].replace('\\', '\\\\').replace('`', '\\`')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{result['name']} - 3D Structure</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #3498db;
        }}
        .main-content {{
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            justify-content: center;
        }}
        .viewer-section {{
            width: 650px;
        }}
        #viewer-container {{
            width: 650px;
            height: 500px;
            position: relative;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: #fafafa;
        }}
        .info-section {{
            width: 350px;
        }}
        .stats {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
        }}
        .stats h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.3em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .stat-row:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            font-weight: 600;
            color: #6c757d;
        }}
        .stat-value {{
            color: #2c3e50;
            font-weight: 500;
        }}
        .quality-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .legend {{
            background: #fff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin-bottom: 20px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 24px;
            height: 24px;
            margin-right: 12px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }}
        .controls {{
            background: #fff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }}
        .controls h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        .btn-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        button {{
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            font-weight: 500;
            transition: all 0.3s ease;
            font-size: 0.9em;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{result['name']} - 3D Structure Prediction</h1>
        
        <div class="main-content">
            <div class="viewer-section">
                <div id="viewer-container"></div>
            </div>
            
            <div class="info-section">
                <div class="stats">
                    <h3>Structure Statistics</h3>
                    <div class="stat-row">
                        <span class="stat-label">Sequence Length</span>
                        <span class="stat-value">{result['length']} residues</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Mean pLDDT</span>
                        <span class="stat-value">{result['mean_plddt']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Min pLDDT</span>
                        <span class="stat-value">{result['min_plddt']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max pLDDT</span>
                        <span class="stat-value">{result['max_plddt']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Quality</span>
                        <span class="stat-value">
                            <span class="quality-badge" style="background: {result['quality_color']}">
                                {result['quality']}
                            </span>
                        </span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Prediction Time</span>
                        <span class="stat-value">{result['prediction_time']:.1f} seconds</span>
                    </div>
                </div>
                
                <div class="legend">
                    <h3>pLDDT Color Legend</h3>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #0053d6;"></div>
                        <span>Very High (>=90) - High confidence</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #65cbf3;"></div>
                        <span>Good (70-90) - Confident</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffdb13;"></div>
                        <span>Moderate (50-70) - Low confidence</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff7d45;"></div>
                        <span>Poor (<50) - Very low confidence</span>
                    </div>
                </div>
                
                <div class="controls">
                    <h3>Controls</h3>
                    <div class="btn-group">
                        <button onclick="startSpin()">Start Spin</button>
                        <button onclick="stopSpin()">Stop Spin</button>
                        <button onclick="resetView()">Reset View</button>
                        <button onclick="setColorScheme('plddt')">Color: pLDDT</button>
                        <button onclick="setColorScheme('rainbow')">Color: Rainbow</button>
                        <button onclick="setColorScheme('white')">Color: White</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Generated by ESMFold Structure Prediction Tool | ISEF 2026 Cold-Adapted Enzyme Project
        </div>
    </div>
    
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        var viewer;
        var pdbData = `{pdb_escaped}`;
        
        // Initialize viewer after page loads
        document.addEventListener('DOMContentLoaded', function() {{
            var element = document.getElementById('viewer-container');
            var config = {{ backgroundColor: 'white' }};
            viewer = $3Dmol.createViewer(element, config);
            
            viewer.addModel(pdbData, 'pdb');
            setColorScheme('plddt');
            viewer.zoomTo();
            viewer.spin(true);
            viewer.render();
        }});
        
        function setColorScheme(scheme) {{
            if (!viewer) return;
            viewer.setStyle({{}}, {{}});
            if (scheme === 'plddt') {{
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorscheme: {{
                            prop: 'b',
                            gradient: 'roygb',
                            min: 50,
                            max: 90
                        }}
                    }}
                }});
            }} else if (scheme === 'rainbow') {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            }} else {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'white'}}}});
            }}
            viewer.render();
        }}
        
        function startSpin() {{
            if (viewer) viewer.spin(true);
        }}
        
        function stopSpin() {{
            if (viewer) viewer.spin(false);
        }}
        
        function resetView() {{
            if (viewer) viewer.zoomTo();
        }}
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"3D visualization saved to: {output_path}")
    
    if auto_open:
        webbrowser.open('file://' + os.path.abspath(output_path))
    
    return output_path


def plot_plddt_profile(result: Dict, save_path: str = None, show: bool = True) -> plt.Figure:
    """
    Create publication-quality pLDDT profile plot.
    
    Parameters
    ----------
    result : dict
        Output from predict_structure_esmfold
    save_path : str, optional
        Path to save figure (e.g., "plddt_profile.png")
    show : bool
        Whether to display the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    plddt = result["plddt_scores"]
    positions = np.arange(1, len(plddt) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create color array based on pLDDT values
    colors = []
    for score in plddt:
        if score >= 90:
            colors.append('#2ecc71')  # Green - very high
        elif score >= 70:
            colors.append('#3498db')  # Blue - good
        elif score >= 50:
            colors.append('#f39c12')  # Orange - moderate
        else:
            colors.append('#e74c3c')  # Red - poor
    
    # Plot bars
    ax.bar(positions, plddt, color=colors, width=1.0, edgecolor='none')
    
    # Add threshold lines
    ax.axhline(y=90, color='#2ecc71', linestyle='--', alpha=0.7, label='Very High (90)')
    ax.axhline(y=70, color='#3498db', linestyle='--', alpha=0.7, label='Good (70)')
    ax.axhline(y=50, color='#f39c12', linestyle='--', alpha=0.7, label='Moderate (50)')
    
    # Add mean line
    ax.axhline(y=result["mean_plddt"], color='black', linestyle='-', 
               linewidth=2, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('pLDDT Score', fontsize=12)
    ax.set_title(f'{result["name"]} - Per-Residue Confidence (Mean: {result["mean_plddt"]:.1f})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(plddt) + 1)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_plddt_interactive(result: Dict, save_path: str = None, 
                           auto_open: bool = True) -> go.Figure:
    """
    Create interactive pLDDT profile with Plotly.
    
    Parameters
    ----------
    result : dict
        Output from predict_structure_esmfold
    save_path : str, optional
        Path to save HTML file
    auto_open : bool
        Whether to open in browser
    
    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure
    """
    plddt = result["plddt_scores"]
    sequence = result["sequence"]
    positions = list(range(1, len(plddt) + 1))
    
    # Create hover text
    hover_text = [f"Position: {i}<br>Residue: {aa}<br>pLDDT: {score:.1f}" 
                  for i, (aa, score) in enumerate(zip(sequence, plddt), 1)]
    
    # Color based on pLDDT
    colors = ['#e74c3c' if s < 50 else '#f39c12' if s < 70 else '#3498db' if s < 90 else '#2ecc71' 
              for s in plddt]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=positions,
        y=plddt,
        marker_color=colors,
        hovertext=hover_text,
        hoverinfo='text',
        name='pLDDT'
    ))
    
    # Add threshold lines
    for y, label, color in [(90, 'Very High', '#2ecc71'), 
                             (70, 'Good', '#3498db'), 
                             (50, 'Moderate', '#f39c12')]:
        fig.add_hline(y=y, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text=f"<b>{result['name']}</b> - Per-Residue Confidence Profile",
            font=dict(size=16)
        ),
        xaxis_title="Residue Position",
        yaxis_title="pLDDT Score",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=500
    )
    
    if save_path is None:
        save_path = f"{result['name']}_plddt_profile.html"
    
    fig.write_html(save_path)
    print(f"Interactive plot saved to: {save_path}")
    
    if auto_open:
        webbrowser.open('file://' + os.path.abspath(save_path))
    
    return fig


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_structures(results: List[Dict], save_path: str = None,
                       auto_open: bool = True) -> go.Figure:
    """
    Compare pLDDT profiles of multiple structures.
    
    Parameters
    ----------
    results : list
        List of results from predict_structure_esmfold
    save_path : str, optional
        Path to save HTML file
    auto_open : bool
        Whether to open in browser
    
    Returns
    -------
    plotly.graph_objects.Figure
        The comparison figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, result in enumerate(results):
        plddt = result["plddt_scores"]
        positions = list(range(1, len(plddt) + 1))
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=plddt,
            mode='lines',
            name=f"{result['name']} (mean: {result['mean_plddt']:.1f})",
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{result['name']}<br>Position: %{{x}}<br>pLDDT: %{{y:.1f}}<extra></extra>"
        ))
    
    # Add threshold lines
    fig.add_hline(y=90, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=70, line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.5)
    
    fig.update_layout(
        title=dict(
            text="<b>pLDDT Profile Comparison</b>",
            font=dict(size=16)
        ),
        xaxis_title="Residue Position",
        yaxis_title="pLDDT Score",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=500,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    
    if save_path is None:
        save_path = "plddt_comparison.html"
    
    fig.write_html(save_path)
    print(f"Comparison plot saved to: {save_path}")
    
    if auto_open:
        webbrowser.open('file://' + os.path.abspath(save_path))
    
    return fig


def plot_comparison_summary(results: List[Dict], save_path: str = None,
                            auto_open: bool = True) -> go.Figure:
    """
    Create summary comparison bar chart.
    
    Parameters
    ----------
    results : list
        List of results from predict_structure_esmfold
    save_path : str, optional
        Path to save HTML file
    auto_open : bool
        Whether to open in browser
    
    Returns
    -------
    plotly.graph_objects.Figure
        The summary figure
    """
    names = [r["name"] for r in results]
    mean_plddt = [r["mean_plddt"] for r in results]
    min_plddt = [r["min_plddt"] for r in results]
    max_plddt = [r["max_plddt"] for r in results]
    
    # Sort by mean pLDDT
    sorted_data = sorted(zip(names, mean_plddt, min_plddt, max_plddt), 
                         key=lambda x: x[1], reverse=True)
    names, mean_plddt, min_plddt, max_plddt = zip(*sorted_data)
    
    # Color based on quality
    colors = ['#2ecc71' if m >= 90 else '#3498db' if m >= 70 else '#f39c12' if m >= 50 else '#e74c3c' 
              for m in mean_plddt]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=names,
        x=mean_plddt,
        orientation='h',
        marker_color=colors,
        error_x=dict(
            type='data',
            symmetric=False,
            array=[max_p - mean for max_p, mean in zip(max_plddt, mean_plddt)],
            arrayminus=[mean - min_p for min_p, mean in zip(min_plddt, mean_plddt)],
            color='rgba(0,0,0,0.3)'
        ),
        text=[f"{m:.1f}" for m in mean_plddt],
        textposition='outside'
    ))
    
    # Add threshold lines
    for x, label in [(90, 'Excellent'), (70, 'Good'), (50, 'Moderate')]:
        fig.add_vline(x=x, line_dash="dash", line_color="gray", opacity=0.5,
                      annotation_text=label, annotation_position="top")
    
    fig.update_layout(
        title=dict(
            text="<b>Structure Quality Comparison</b>",
            font=dict(size=16)
        ),
        xaxis_title="Mean pLDDT Score",
        xaxis_range=[0, 105],
        template="plotly_white",
        height=max(300, len(names) * 50),
        showlegend=False
    )
    
    if save_path is None:
        save_path = "quality_comparison.html"
    
    fig.write_html(save_path)
    print(f"Summary plot saved to: {save_path}")
    
    if auto_open:
        webbrowser.open('file://' + os.path.abspath(save_path))
    
    return fig


def plot_plddt_difference(wt_result: Dict, mut_result: Dict, 
                          save_path: str = None, auto_open: bool = True) -> Tuple[go.Figure, float]:
    """
    Plot pLDDT difference between wild-type and mutant.
    
    Parameters
    ----------
    wt_result : dict
        Wild-type result from predict_structure_esmfold
    mut_result : dict
        Mutant result from predict_structure_esmfold
    save_path : str, optional
        Path to save HTML file
    auto_open : bool
        Whether to open in browser
    
    Returns
    -------
    tuple
        (Plotly figure, mean difference)
    """
    wt_plddt = wt_result["plddt_scores"]
    mut_plddt = mut_result["plddt_scores"]
    
    # Handle length differences
    min_len = min(len(wt_plddt), len(mut_plddt))
    diff = mut_plddt[:min_len] - wt_plddt[:min_len]
    positions = list(range(1, min_len + 1))
    
    # Color based on improvement/degradation
    colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=positions,
        y=diff,
        marker_color=colors,
        hovertemplate="Position: %{x}<br>Δ pLDDT: %{y:+.1f}<extra></extra>"
    ))
    
    fig.add_hline(y=0, line_color="black", line_width=2)
    
    mean_diff = np.mean(diff)
    
    fig.update_layout(
        title=dict(
            text=f"<b>pLDDT Difference: {mut_result['name']} vs {wt_result['name']}</b><br>" +
                 f"<sup>Mean Δ: {mean_diff:+.2f} | Green = Improvement, Red = Degradation</sup>",
            font=dict(size=14)
        ),
        xaxis_title="Residue Position",
        yaxis_title="Δ pLDDT (Mutant - Wild-Type)",
        template="plotly_white",
        height=400
    )
    
    if save_path is None:
        save_path = f"plddt_difference_{mut_result['name']}.html"
    
    fig.write_html(save_path)
    print(f"Difference plot saved to: {save_path}")
    
    if auto_open:
        webbrowser.open('file://' + os.path.abspath(save_path))
    
    return fig, mean_diff


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def generate_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    Generate summary DataFrame for all analyzed structures.
    
    Parameters
    ----------
    results : list
        List of results from predict_structure_esmfold
    
    Returns
    -------
    pandas.DataFrame
        Summary table
    """
    data = []
    for r in results:
        plddt = r["plddt_scores"]
        excellent = np.sum(plddt >= 90)
        good = np.sum((plddt >= 70) & (plddt < 90))
        moderate = np.sum((plddt >= 50) & (plddt < 70))
        poor = np.sum(plddt < 50)
        
        data.append({
            "Name": r["name"],
            "Length": r["length"],
            "Mean_pLDDT": round(r["mean_plddt"], 1),
            "Min_pLDDT": round(r["min_plddt"], 1),
            "Max_pLDDT": round(r["max_plddt"], 1),
            "Std_pLDDT": round(r["std_plddt"], 1),
            "Quality": r["quality"],
            "Excellent_pct": round(100 * excellent / r["length"], 1),
            "Good_pct": round(100 * good / r["length"], 1),
            "Moderate_pct": round(100 * moderate / r["length"], 1),
            "Poor_pct": round(100 * poor / r["length"], 1)
        })
    
    df = pd.DataFrame(data)
    return df


def print_summary_report(results: List[Dict]):
    """
    Print formatted summary report to console.
    
    Parameters
    ----------
    results : list
        List of results from predict_structure_esmfold
    """
    print()
    print("=" * 80)
    print("           ESMFOLD STRUCTURAL VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    # Sort by mean pLDDT
    sorted_results = sorted(results, key=lambda x: x["mean_plddt"], reverse=True)
    
    for rank, r in enumerate(sorted_results, 1):
        plddt = r["plddt_scores"]
        excellent_pct = 100 * np.sum(plddt >= 90) / len(plddt)
        
        print(f"Rank {rank}: {r['name']}")
        print(f"  Sequence Length:  {r['length']} residues")
        print(f"  Mean pLDDT:       {r['mean_plddt']:.1f} ({r['quality']})")
        print(f"  Range:            {r['min_plddt']:.1f} - {r['max_plddt']:.1f}")
        print(f"  High Confidence:  {excellent_pct:.1f}% residues with pLDDT ≥ 90")
        print(f"  Prediction Time:  {r['prediction_time']:.1f} seconds")
        print()
    
    print("=" * 80)
    print("INTERPRETATION GUIDE:")
    print("  pLDDT ≥ 90:  Very high confidence - well-folded regions")
    print("  pLDDT 70-90: Good confidence - likely correct fold")
    print("  pLDDT 50-70: Low confidence - may have local errors")
    print("  pLDDT < 50:  Very low confidence - likely disordered")
    print("=" * 80)


def save_pdb_file(result: Dict, output_dir: str = ".") -> str:
    """
    Save PDB structure to file.
    
    Parameters
    ----------
    result : dict
        Output from predict_structure_esmfold
    output_dir : str
        Output directory
    
    Returns
    -------
    str
        Path to saved PDB file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{result['name']}_ESMFold.pdb")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result['pdb_string'])
    
    print(f"PDB file saved to: {filename}")
    return filename


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    """Main program - single run, saves all outputs to esmfold_results folder."""

    parser = argparse.ArgumentParser(description='ESMFold Protein Structure Prediction')
    parser.add_argument('--sequence', type=str, default='',
                        help='Protein amino acid sequence')
    parser.add_argument('--name', type=str, default='Protein',
                        help='Protein name (default: Protein)')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("    ESMFold Protein Structure Prediction Tool")
    print("    (API-based - No GPU Required)")
    print("=" * 70)
    print()

    output_dir = "esmfold_results"
    os.makedirs(output_dir, exist_ok=True)

    # Get sequence from argument or interactive input
    if args.sequence.strip():
        sequence = ''.join(args.sequence.strip().split())
    else:
        print("Enter protein sequence (paste and press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        sequence = ''.join(''.join(lines).split())

    if not sequence:
        print("No sequence provided. Exiting.")
        return

    # Get name from argument or interactive input
    if args.name.strip() and args.sequence.strip():
        name = args.name.strip()
    else:
        name = input("Enter protein name (or press Enter for default): ").strip()
        if not name:
            name = "Protein"
    
    # Predict structure
    result = predict_structure_esmfold(sequence, name)
    
    if result:
        # Save all outputs (no browser opening)
        pdb_path = save_pdb_file(result, output_dir)
        
        html_path = os.path.join(output_dir, f"{name}_3d.html")
        create_3d_html_viewer(result, html_path, auto_open=False)
        
        png_path = os.path.join(output_dir, f"{name}_plddt.png")
        plot_plddt_profile(result, png_path, show=False)
        
        interactive_path = os.path.join(output_dir, f"{name}_plddt_interactive.html")
        plot_plddt_interactive(result, interactive_path, auto_open=False)
        
        # Print summary
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Protein Name:     {result['name']}")
        print(f"  Sequence Length:  {result['length']} residues")
        print(f"  Mean pLDDT:       {result['mean_plddt']:.1f} ({result['quality']})")
        print(f"  Min pLDDT:        {result['min_plddt']:.1f}")
        print(f"  Max pLDDT:        {result['max_plddt']:.1f}")
        print()
        print("OUTPUT FILES:")
        print(f"  PDB Structure:    {pdb_path}")
        print(f"  3D Viewer:        {html_path}")
        print(f"  pLDDT Plot (PNG): {png_path}")
        print(f"  Interactive Plot: {interactive_path}")
        print("=" * 70)
    else:
        print("Structure prediction failed.")


if __name__ == "__main__":
    main()
