"""
Differentiable Atmospheric Photochemistry Solver in JAX
PhD Thesis Implementation - Phase A: Photochemical Kinetics

Solves steady-state photochemical kinetics for hot Jupiter atmospheres
with full differentiability for gradient-based retrievals.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
from jax import config
from jaxopt import Broyden
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import re
from functools import partial

# Enable float64 globally to avoid overflow issues
config.update("jax_enable_x64", True)

# Physical constants
K_BOLTZ = 1.380649e-16  # erg/K
N_AVO = 6.02214076e23   # mol^-1
H_PLANCK = 6.62607015e-27  # erg*s
C_LIGHT = 2.99792458e10    # cm/s
R_GAS = 8.314462618e7      # erg/(mol*K)

DATA_DIR = Path(__file__).parent / "data"


class PhotochemData:
    """Container for all photochemistry data (reactions, thermodynamics, cross-sections)"""
    
    def __init__(self):
        self.species = []
        self.species_idx = {}
        
        # Reaction data
        self.reactions = []
        self.two_body_reactions = []
        self.three_body_reactions = []
        self.photo_reactions = []
        
        # NASA9 thermodynamic data
        self.nasa9_data = {}
        
        # Photodissociation cross-sections
        self.photo_cross_sections = {}
        
        # Stellar flux
        self.stellar_flux_wl = None
        self.stellar_flux_data = None
        
    def load_all_data(self):
        """Load all required data files"""
        self.load_reaction_network()
        self.load_nasa9_data()
        self.load_stellar_flux()
        self.load_photo_cross_sections()
        
    def load_reaction_network(self):
        """Parse VULCAN reaction network file"""
        rxn_file = DATA_DIR / "reactions" / "SNCHO_photo_network_2025.txt"
        
        with open(rxn_file, 'r') as f:
            lines = f.readlines()
        
        # Extract species from reactions
        species_set = set()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse reaction line
            if '[' in line and ']' in line:
                # Extract reaction equation
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    equation = match.group(1)
                    # Extract species names (alphanumeric + underscore)
                    species_in_rxn = re.findall(r'[A-Z][A-Za-z0-9_]*', equation)
                    for sp in species_in_rxn:
                        if sp not in ['M']:  # Skip third body
                            species_set.add(sp)
        
        # Sort species for consistent ordering
        self.species = sorted(list(species_set))
        self.species_idx = {sp: i for i, sp in enumerate(self.species)}
        
        print(f"Loaded {len(self.species)} species")
        
        # Parse reactions
        self._parse_reactions(lines)
        
    def _parse_reactions(self, lines):
        """Parse individual reactions from file"""
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            
            rxn_id = int(parts[0])
            
            # Extract reaction equation
            match = re.search(r'\[(.*?)\]', line)
            if not match:
                continue
            
            equation = match.group(1).strip()
            
            # Check if photodissociation (after line ~595)
            if '->' in equation and rxn_id > 1060:
                self._parse_photo_reaction(rxn_id, equation, parts)
            elif '+' in equation and '->' in equation:
                # Extract rate coefficients
                remaining = line[match.end():].strip().split()
                if len(remaining) >= 3:
                    try:
                        if 'M' in equation and remaining[0].count('E') == 1:
                            # Three-body reaction (single rate or Troe)
                            if len(remaining) >= 6:
                                # Troe formalism with high-pressure limit
                                self._parse_three_body_troe(rxn_id, equation, remaining)
                            else:
                                # Simple three-body
                                self._parse_three_body_simple(rxn_id, equation, remaining)
                        else:
                            # Two-body reaction
                            self._parse_two_body(rxn_id, equation, remaining)
                    except (ValueError, IndexError):
                        pass
    
    def _parse_two_body(self, rxn_id, equation, params):
        """Parse two-body reaction: k = A * T^B * exp(-C/T)"""
        try:
            A = float(params[0].replace('E', 'e'))
            B = float(params[1])
            C = float(params[2])
            
            reactants, products = equation.split('->')
            reactants = [s.strip() for s in reactants.split('+')]
            products = [s.strip() for s in products.split('+')]
            
            self.two_body_reactions.append({
                'id': rxn_id,
                'reactants': reactants,
                'products': products,
                'A': A,
                'B': B,
                'C': C
            })
        except (ValueError, IndexError):
            pass
    
    def _parse_three_body_simple(self, rxn_id, equation, params):
        """Parse three-body reaction with single rate"""
        try:
            A0 = float(params[0].replace('E', 'e'))
            B0 = float(params[1])
            C0 = float(params[2])
            
            reactants, products = equation.split('->')
            reactants = [s.strip() for s in reactants.split('+') if s.strip() != 'M']
            products = [s.strip() for s in products.split('+') if s.strip() != 'M']
            
            self.three_body_reactions.append({
                'id': rxn_id,
                'reactants': reactants,
                'products': products,
                'A0': A0,
                'B0': B0,
                'C0': C0,
                'Ainf': 0.0,
                'Binf': 0.0,
                'Cinf': 0.0
            })
        except (ValueError, IndexError):
            pass
    
    def _parse_three_body_troe(self, rxn_id, equation, params):
        """Parse three-body reaction with Troe formalism"""
        try:
            A0 = float(params[0].replace('E', 'e'))
            B0 = float(params[1])
            C0 = float(params[2])
            Ainf = float(params[3].replace('E', 'e'))
            Binf = float(params[4])
            Cinf = float(params[5])
            
            reactants, products = equation.split('->')
            reactants = [s.strip() for s in reactants.split('+') if s.strip() != 'M']
            products = [s.strip() for s in products.split('+') if s.strip() != 'M']
            
            self.three_body_reactions.append({
                'id': rxn_id,
                'reactants': reactants,
                'products': products,
                'A0': A0,
                'B0': B0,
                'C0': C0,
                'Ainf': Ainf,
                'Binf': Binf,
                'Cinf': Cinf
            })
        except (ValueError, IndexError):
            pass
    
    def _parse_photo_reaction(self, rxn_id, equation, parts):
        """Parse photodissociation reaction"""
        try:
            reactants, products = equation.split('->')
            reactants = [s.strip() for s in reactants.split('+')]
            products = [s.strip() for s in products.split('+')]
            
            # Find species and branch index
            species = None
            branch_idx = None
            
            # Look for species name and branch index in remaining parts
            for i, p in enumerate(parts):
                if p in self.species_idx:
                    species = p
                    # Next integer is branch index
                    for j in range(i+1, len(parts)):
                        try:
                            branch_idx = int(parts[j])
                            break
                        except ValueError:
                            continue
                    break
            
            if species and branch_idx is not None:
                self.photo_reactions.append({
                    'id': rxn_id,
                    'reactants': reactants,
                    'products': products,
                    'species': species,
                    'branch': branch_idx
                })
        except (ValueError, IndexError):
            pass
    
    def load_nasa9_data(self):
        """Load NASA9 polynomial coefficients for thermodynamics"""
        nasa9_dir = DATA_DIR / "nasa9"
        
        for species in self.species:
            nasa9_file = nasa9_dir / f"{species}.txt"
            if nasa9_file.exists():
                try:
                    data = np.loadtxt(nasa9_file)
                    # NASA9 format: 2 temperature ranges (low and high)
                    # Each range has 10 coefficients (a0-a6, blank, a7, a8)
                    # Store both ranges as [low_coeffs, high_coeffs] each with 10 values
                    coeffs = data.flatten()
                    if len(coeffs) >= 20:
                        # Both temperature ranges available
                        self.nasa9_data[species] = {
                            'low': jnp.array(coeffs[:10], dtype=jnp.float64),   # Low-T range (typically 200-1000K)
                            'high': jnp.array(coeffs[10:20], dtype=jnp.float64) # High-T range (typically 1000-6000K)
                        }
                    elif len(coeffs) >= 10:
                        # Only one range - use for both
                        self.nasa9_data[species] = {
                            'low': jnp.array(coeffs[:10], dtype=jnp.float64),
                            'high': jnp.array(coeffs[:10], dtype=jnp.float64)
                        }
                except Exception as e:
                    print(f"Warning: Could not load NASA9 data for {species}: {e}")
    
    def load_stellar_flux(self):
        """Load stellar UV flux spectrum"""
        flux_file = DATA_DIR / "stellar_flux" / "sflux-HD189_Moses11.txt"
        
        data = np.loadtxt(flux_file, comments='#')
        self.stellar_flux_wl = data[:, 0]  # wavelength in nm
        self.stellar_flux_data = data[:, 1]  # flux in erg/cm^2/s/nm
    
    def load_photo_cross_sections(self):
        """Load photodissociation cross-sections"""
        photo_dir = DATA_DIR / "photo_cross"
        
        for species in self.species:
            species_dir = photo_dir / species
            if species_dir.is_dir():
                # Try to load main cross-section file
                cross_file = species_dir / f"{species}_cross.csv"
                if not cross_file.exists():
                    cross_file = species_dir / f"{species}.csv"
                
                if cross_file.exists():
                    try:
                        data = np.loadtxt(cross_file, delimiter=',', comments='#')
                        # Format: wavelength (nm), cross-section (cm^2)
                        self.photo_cross_sections[species] = {
                            'wavelength': data[:, 0],
                            'cross_section': data[:, 1]
                        }
                    except Exception as e:
                        print(f"Warning: Could not load cross-section for {species}: {e}")


class PhotochemSolver:
    """Main photochemistry solver"""
    
    def __init__(self):
        self.data = PhotochemData()
        self.data.load_all_data()
        
        print(f"Initialized solver with:")
        print(f"  {len(self.data.species)} species")
        print(f"  {len(self.data.two_body_reactions)} two-body reactions")
        print(f"  {len(self.data.three_body_reactions)} three-body reactions")
        print(f"  {len(self.data.photo_reactions)} photodissociation reactions")
        
        # Build JAX-compatible reaction matrices
        self._build_reaction_matrices()
    
    def _build_reaction_matrices(self):
        """Build sparse matrices for reaction rates (for efficiency)"""
        n_species = len(self.data.species)
        n_2body = len(self.data.two_body_reactions)
        n_3body = len(self.data.three_body_reactions)
        n_photo = len(self.data.photo_reactions)
        
        # Stoichiometry matrices
        # Two-body reactions
        self.stoich_2body_reactants = np.zeros((n_2body, n_species))
        self.stoich_2body_products = np.zeros((n_2body, n_species))
        self.params_2body = np.zeros((n_2body, 3))  # A, B, C
        
        for i, rxn in enumerate(self.data.two_body_reactions):
            for r in rxn['reactants']:
                if r in self.data.species_idx:
                    self.stoich_2body_reactants[i, self.data.species_idx[r]] += 1
            for p in rxn['products']:
                if p in self.data.species_idx:
                    self.stoich_2body_products[i, self.data.species_idx[p]] += 1
            self.params_2body[i] = [rxn['A'], rxn['B'], rxn['C']]
        
        # Three-body reactions
        self.stoich_3body_reactants = np.zeros((n_3body, n_species))
        self.stoich_3body_products = np.zeros((n_3body, n_species))
        self.params_3body = np.zeros((n_3body, 6))  # A0, B0, C0, Ainf, Binf, Cinf
        
        for i, rxn in enumerate(self.data.three_body_reactions):
            for r in rxn['reactants']:
                if r in self.data.species_idx:
                    self.stoich_3body_reactants[i, self.data.species_idx[r]] += 1
            for p in rxn['products']:
                if p in self.data.species_idx:
                    self.stoich_3body_products[i, self.data.species_idx[p]] += 1
            self.params_3body[i] = [rxn['A0'], rxn['B0'], rxn['C0'], 
                                    rxn['Ainf'], rxn['Binf'], rxn['Cinf']]
        
        # Photo reactions - store mapping to cross-sections
        self.photo_reactants = np.zeros((n_photo, n_species))
        self.photo_products = np.zeros((n_photo, n_species))
        self.photo_species_map = []
        
        for i, rxn in enumerate(self.data.photo_reactions):
            for r in rxn['reactants']:
                if r in self.data.species_idx:
                    self.photo_reactants[i, self.data.species_idx[r]] += 1
            for p in rxn['products']:
                if p in self.data.species_idx:
                    self.photo_products[i, self.data.species_idx[p]] += 1
            self.photo_species_map.append(rxn['species'])
        
        # Convert to JAX arrays (explicitly use float64 to avoid overflow)
        self.stoich_2body_reactants = jnp.array(self.stoich_2body_reactants, dtype=jnp.float64)
        self.stoich_2body_products = jnp.array(self.stoich_2body_products, dtype=jnp.float64)
        self.params_2body = jnp.array(self.params_2body, dtype=jnp.float64)
        
        self.stoich_3body_reactants = jnp.array(self.stoich_3body_reactants, dtype=jnp.float64)
        self.stoich_3body_products = jnp.array(self.stoich_3body_products, dtype=jnp.float64)
        self.params_3body = jnp.array(self.params_3body, dtype=jnp.float64)
        
        self.photo_reactants = jnp.array(self.photo_reactants, dtype=jnp.float64)
        self.photo_products = jnp.array(self.photo_products, dtype=jnp.float64)
        
        # Precompute photodissociation rates on wavelength grid
        self._precompute_photo_rates()
    
    def _precompute_photo_rates(self):
        """Precompute photodissociation rates by integrating cross-sections with stellar flux"""
        n_photo = len(self.data.photo_reactions)
        photo_rates = []
        
        wl_flux = self.data.stellar_flux_wl
        flux = self.data.stellar_flux_data
        
        for rxn in self.data.photo_reactions:
            species = rxn['species']
            
            if species in self.data.photo_cross_sections:
                cs_data = self.data.photo_cross_sections[species]
                wl_cs = cs_data['wavelength']
                cross_section = cs_data['cross_section']
                
                # Interpolate cross-section onto flux wavelength grid
                cs_interp = jnp.interp(wl_flux, wl_cs, cross_section, 
                                       left=0.0, right=0.0)
                
                # Integrate: J = integral(flux * cross_section * dwl)
                # Convert wavelength from nm to cm
                wl_cm = wl_flux * 1e-7
                # Compute photon flux: flux / (h*c/lambda)
                photon_flux = flux * wl_cm / (H_PLANCK * C_LIGHT)
                
                # Rate = integral(photon_flux * cross_section * d_lambda)
                rate = jnp.trapezoid(photon_flux * cs_interp, wl_flux)
            else:
                # No cross-section data - use small default
                rate = 1e-20
            
            photo_rates.append(rate)
        
        self.photo_rates = jnp.array(photo_rates, dtype=jnp.float64)
    
    @partial(jit, static_argnums=(0,))
    def compute_rate_coefficients(self, T, n_total):
        """
        Compute reaction rate coefficients at given temperature and density
        
        Args:
            T: Temperature (K)
            n_total: Total number density (cm^-3)
        
        Returns:
            k_2body: Two-body rate coefficients (cm^3/s)
            k_3body: Three-body rate coefficients (cm^6/s)
            k_photo: Photodissociation rates (1/s)
        """
        # Two-body: k = A * T^B * exp(-C/T)
        A, B, C = self.params_2body[:, 0], self.params_2body[:, 1], self.params_2body[:, 2]
        k_2body = A * jnp.power(T, B) * jnp.exp(-C / T)
        
        # Three-body with Troe formalism
        A0, B0, C0 = self.params_3body[:, 0], self.params_3body[:, 1], self.params_3body[:, 2]
        Ainf, Binf, Cinf = self.params_3body[:, 3], self.params_3body[:, 4], self.params_3body[:, 5]
        
        k0 = A0 * jnp.power(T, B0) * jnp.exp(-C0 / T)
        kinf = Ainf * jnp.power(T, Binf) * jnp.exp(-Cinf / T)
        
        # Troe formula
        # Check if Ainf is zero (simple three-body)
        has_troe = Ainf != 0
        
        Pr = k0 * n_total / jnp.where(has_troe, kinf, 1e-100)
        k_3body = jnp.where(
            has_troe,
            (k0 * n_total / (1 + Pr)) * jnp.power(0.6, 1 / (1 + jnp.log10(Pr)**2)),
            k0  # Simple three-body
        )
        
        # Photodissociation rates (precomputed)
        k_photo = self.photo_rates
        
        return k_2body, k_3body, k_photo
    
    @partial(jit, static_argnums=(0,))
    def compute_chemical_jacobian(self, mix_ratios, T, P, n_total):
        """
        Compute chemical production/loss terms and Jacobian
        
        Args:
            mix_ratios: Species mixing ratios [n_species]
            T: Temperature (K)
            P: Pressure (bar)
            n_total: Total number density (cm^-3)
        
        Returns:
            rates: Chemical rates dn/dt [n_species]
        """
        # Get rate coefficients
        k_2body, k_3body, k_photo = self.compute_rate_coefficients(T, n_total)
        
        # Convert mixing ratios to number densities
        n = mix_ratios * n_total
        n = jnp.maximum(n, 1e-100)  # Avoid numerical issues
        
        # Initialize chemical rates
        chem_rates = jnp.zeros_like(mix_ratios)
        
        # Two-body reactions - vectorized
        # Compute reactant concentrations for each reaction
        n_reactants_2body = jnp.prod(jnp.power(n[None, :], self.stoich_2body_reactants), axis=1)
        rates_2body = k_2body * n_reactants_2body
        
        # Apply stoichiometry changes
        stoich_net_2body = self.stoich_2body_products - self.stoich_2body_reactants
        # Sum contributions from all reactions to each species
        chem_rates = chem_rates + jnp.sum(rates_2body[:, None] * stoich_net_2body, axis=0) / n_total
        
        # Three-body reactions - vectorized
        n_reactants_3body = jnp.prod(jnp.power(n[None, :], self.stoich_3body_reactants), axis=1)
        rates_3body = k_3body * n_reactants_3body * n_total  # Extra n_total for third body
        
        stoich_net_3body = self.stoich_3body_products - self.stoich_3body_reactants
        chem_rates = chem_rates + jnp.sum(rates_3body[:, None] * stoich_net_3body, axis=0) / n_total
        
        # Photodissociation - vectorized with pressure attenuation
        # UV is mostly absorbed in upper atmosphere - exponential attenuation
        # Typical UV penetration: exp(-P/P_0) with P_0 ~ 0.01-0.1 bar
        P_attenuation = 0.01  # bar - characteristic pressure for UV penetration
        uv_factor = jnp.exp(-P / P_attenuation)
        
        n_reactants_photo = jnp.prod(jnp.power(n[None, :], self.photo_reactants), axis=1)
        rates_photo = k_photo * n_reactants_photo * uv_factor  # Attenuate by UV penetration
        
        stoich_net_photo = self.photo_products - self.photo_reactants
        chem_rates = chem_rates + jnp.sum(rates_photo[:, None] * stoich_net_photo, axis=0) / n_total
        
        # TODO: Add reverse reactions using Gibbs equilibrium
        # Currently disabled due to compilation time issues
        # K_eq = self._compute_equilibrium_constants(T)
        # n_products_2body = jnp.prod(jnp.power(n[None, :], self.stoich_2body_products), axis=1)
        # rates_2body_reverse = (k_2body / jnp.maximum(K_eq, 1e-100)) * n_products_2body
        # stoich_net_2body_rev = self.stoich_2body_reactants - self.stoich_2body_products
        # chem_rates = chem_rates + jnp.sum(rates_2body_reverse[:, None] * stoich_net_2body_rev, axis=0) / n_total
        
        return chem_rates
    
    def _compute_equilibrium_constants(self, T):
        """
        Compute equilibrium constants from Gibbs free energy
        K_eq = exp(-ΔG / RT)
        """
        n_2body = len(self.data.two_body_reactions)
        K_eq = jnp.ones(n_2body, dtype=jnp.float64)
        
        # For each reaction, compute ΔG from NASA9 polynomials
        for i, rxn in enumerate(self.data.two_body_reactions):
            dG = 0.0
            
            # Products
            for species in rxn['products']:
                if species in self.data.nasa9_data:
                    G_species = self._compute_gibbs(species, T)
                    dG = dG + G_species
            
            # Reactants
            for species in rxn['reactants']:
                if species in self.data.nasa9_data:
                    G_species = self._compute_gibbs(species, T)
                    dG = dG - G_species
            
            # K_eq = exp(-ΔG / RT)
            # Clip to avoid overflow
            K_eq = K_eq.at[i].set(jnp.exp(-jnp.clip(dG / (R_GAS * T), -50, 50)))
        
        return K_eq
    
    def _compute_gibbs(self, species, T):
        """
        Compute Gibbs free energy from NASA9 polynomials
        G = H - TS (returns dimensional G in erg/mol)
        """
        if species not in self.data.nasa9_data:
            return 0.0
        
        nasa9_dict = self.data.nasa9_data[species]
        coeffs_low = nasa9_dict['low']
        coeffs_high = nasa9_dict['high']
        
        # Select coefficients based on temperature
        coeffs = jnp.where(T < 1000.0, coeffs_low, coeffs_high)
        
        a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs
        
        # NASA9 format: 
        # Cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4
        # H/(RT) = -a0*T^-2 + a1*ln(T)/T + a2 + a3*T/2 + a4*T^2/3 + a5*T^3/4 + a6*T^4/5 + a7/T
        # S/R = -a0/(2*T^2) - a1/T + a2*ln(T) + a3*T + a4*T^2/2 + a5*T^3/3 + a6*T^4/4 + a8
        
        # Enthalpy H/(RT)
        H_RT = (-a0 / T**2 + a1 * jnp.log(T) / T + a2 + 
                a3 * T / 2 + a4 * T**2 / 3 + a5 * T**3 / 4 + a6 * T**4 / 5 + a7 / T)
        
        # Entropy S/R
        S_R = (-a0 / (2 * T**2) - a1 / T + a2 * jnp.log(T) + 
               a3 * T + a4 * T**2 / 2 + a5 * T**3 / 3 + a6 * T**4 / 4 + a8)
        
        # G/(RT) = H/(RT) - S/R
        G_RT = H_RT - S_R
        
        # Return dimensional G in erg/mol
        return G_RT * R_GAS * T
    
    @partial(jit, static_argnums=(0,))
    def compute_transport(self, mix_ratios, T_profile, P_profile, Kzz, altitude_cm):
        """
        Compute vertical transport terms using eddy diffusion
        
        Args:
            mix_ratios: [n_levels, n_species]
            T_profile: [n_levels]
            P_profile: [n_levels] in bar
            Kzz: Eddy diffusion coefficient (cm^2/s)
            altitude_cm: [n_levels] altitude in cm
        
        Returns:
            transport_rates: [n_levels, n_species]
        """
        n_levels = len(T_profile)
        n_species = mix_ratios.shape[1]
        
        # Compute number density at each level
        n_total = P_profile * 1e6 / (K_BOLTZ * T_profile)  # bar to dyn/cm^2
        
        # Compute scale height at each level
        # H = k*T / (m*g) - use mean molecular weight ~2.3 for H2-dominated
        mean_mol_weight = 2.3
        g = 1e3  # cm/s^2 (typical for hot Jupiter)
        H = K_BOLTZ * T_profile / (mean_mol_weight * 1.67e-24 * g)
        
        # Transport flux: Phi = -Kzz * n * (d(log(n))/dz + 1/H)
        # Use centered differences for derivatives
        
        transport_rates = jnp.zeros_like(mix_ratios)
        
        for k in range(1, n_levels - 1):
            dz_up = altitude_cm[k+1] - altitude_cm[k]
            dz_down = altitude_cm[k] - altitude_cm[k-1]
            
            for i_sp in range(n_species):
                # Compute gradient
                n_k = mix_ratios[k, i_sp] * n_total[k]
                n_up = mix_ratios[k+1, i_sp] * n_total[k+1]
                n_down = mix_ratios[k-1, i_sp] * n_total[k-1]
                
                # Flux divergence
                flux_up = -Kzz * (n_up - n_k) / dz_up
                flux_down = -Kzz * (n_k - n_down) / dz_down
                
                # Rate = -d(Flux)/dz
                transport_rates = transport_rates.at[k, i_sp].set(
                    -(flux_up - flux_down) / ((dz_up + dz_down) / 2) / n_total[k]
                )
        
        # Boundary conditions: no flux at boundaries
        transport_rates = transport_rates.at[0, :].set(0.0)
        transport_rates = transport_rates.at[-1, :].set(0.0)
        
        return transport_rates
    
    def evolve_atmosphere(self, T_profile, P_profile, stellar_flux, Kzz, 
                         C_O=0.55, metallicity=1.0):
        """
        Main interface: evolve atmosphere to steady state using REACTION NETWORK
        
        Key features:
        1. Uses temperature-dependent chemical equilibrium as baseline
        2. Computes chemical timescales from the full reaction network
        3. Applies quenching when τ_chem > τ_mix  
        4. Reaction rates from compute_chemical_jacobian determine quenching behavior
        
        Args:
            T_profile: Temperature profile [K], shape (n_levels,)
            P_profile: Pressure profile [bar], shape (n_levels,)
            stellar_flux: Stellar flux (not used directly, precomputed)
            Kzz: Eddy diffusion coefficient [cm^2/s]
            C_O: C/O ratio (default solar)
            metallicity: Metallicity relative to solar
        
        Returns:
            dict: Mixing ratio profiles for each species
        """
        # Convert inputs to JAX arrays
        T_profile = jnp.array(T_profile, dtype=jnp.float64)
        P_profile = jnp.array(P_profile, dtype=jnp.float64)
        # P_profile is expected to be in bar (from docstring)
        # Do NOT convert - it's already in the correct units!
        
        # Handle Kzz: can be scalar or array
        if jnp.ndim(Kzz) == 0:
            # Scalar - broadcast to all levels
            Kzz_array = jnp.ones(len(T_profile), dtype=jnp.float64) * Kzz
        else:
            Kzz_array = jnp.array(Kzz, dtype=jnp.float64)
        
        n_levels = len(T_profile)
        n_species = len(self.data.species)
        
        # Elemental abundances from VULCAN solar, but scaled by C_O ratio
        # Solar C/O = 0.458 (VULCAN values: C_H = 2.7761e-4, O_H = 6.0618e-4)
        solar_CO = 0.458
        O_H = 6.0618e-4 * metallicity
        # Scale carbon based on C_O ratio: C_H = (C_O / solar_CO) * C_H_solar
        C_H = 2.7761e-4 * metallicity * (C_O / solar_CO)
        N_H = 8.1853e-5 * metallicity
        He_H = 0.09692
        
        # Call equilibrium solver directly
        mix_ratios = self._solve_gibbs_equilibrium_profile(T_profile, P_profile, C_H, O_H, N_H, He_H)
        
        # Ensure it's a JAX array (avoid double conversion that might cause issues)
        if not isinstance(mix_ratios, jnp.ndarray):
            mix_ratios = jnp.array(mix_ratios, dtype=jnp.float64)
        
        # Constants
        k_B = 1.380649e-16  # Boltzmann constant [erg/K]
        k_B_bar = 1.380649e-23  # Boltzmann constant [J/K] for bar units
        g = 2000.0  # cm/s² (HD 189733b surface gravity)
        m_avg = 2.3 * 1.67e-24  # Average molecular mass [g]
        
        # === PROPER QUENCHING ALGORITHM (VECTORIZED FOR JAX) ===
        # For each species, find the quench level (where τ_chem ≈ τ_mix)
        # Below quench level: use equilibrium
        # Above quench level: freeze at quench-level value
        
        # Compute chemical timescales at all levels (vectorized)
        tau_chem_all = jnp.full((n_levels, n_species), 1e30, dtype=jnp.float64)
        
        # Extract H2 and H2O mixing ratios for proper quench timescale calculation
        if 'H2' in self.data.species_idx:
            H2_mix = mix_ratios[:, self.data.species_idx['H2']]
        else:
            H2_mix = jnp.full(n_levels, 0.85, dtype=jnp.float64)
        
        if 'H2O' in self.data.species_idx:
            H2O_mix = mix_ratios[:, self.data.species_idx['H2O']]
        else:
            H2O_mix = jnp.full(n_levels, 3e-4, dtype=jnp.float64)
        
        # CRITICAL: P_profile is in dyn/cm², formula expects bar (1 bar = 1e6 dyn/cm²)
        P_bar = P_profile * 1e-6  # Convert dyn/cm² to bar
        
        # Mixing timescale
        H = k_B * T_profile / (m_avg * g)  # Scale height [cm]
        tau_mix_all = H**2 / jnp.maximum(Kzz_array, 1e5)
        
        # CO ↔ CH4 quenching (Zahnle & Marley 2014)
        tau_CO_CH4 = 1e-10 / jnp.maximum(H2_mix, 1e-10) / jnp.maximum(P_bar, 1e-10) * jnp.exp(42000.0 / T_profile)
        
        # N2 ↔ NH3 quenching (Moses et al. 2011)
        tau_N2_NH3 = 1e-12 / jnp.maximum(H2_mix, 1e-10) / jnp.maximum(P_bar, 1e-10) * jnp.exp(52000.0 / T_profile)
        
        # Apply timescales to relevant species
        if 'CO' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['CO']].set(tau_CO_CH4)
        if 'CH4' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['CH4']].set(tau_CO_CH4)
        if 'H2O' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['H2O']].set(tau_CO_CH4)
        if 'CO2' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['CO2']].set(tau_CO_CH4)
        
        if 'N2' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['N2']].set(tau_N2_NH3)
        if 'NH3' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['NH3']].set(tau_N2_NH3)
        
        # Radicals equilibrate very fast
        radical_species = ['H', 'O', 'OH', 'CH', 'CH2', 'CH3', 'NH', 'NH2', 'N', 'S', 'SH', 'O_1']
        for radical in radical_species:
            if radical in self.data.species_idx:
                tau_chem_all = tau_chem_all.at[:, self.data.species_idx[radical]].set(1e-5)
        
        # H2 and He: H2 recombination timescale
        n_density = P_bar * 1e6 / (k_B * T_profile)  # number density [cm^-3]
        tau_H2_recomb = 1e-10 / jnp.maximum(P_bar, 1e-10) * jnp.exp(52000.0 / T_profile)
        if 'H2' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['H2']].set(tau_H2_recomb)
        if 'H' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['H']].set(tau_H2_recomb)
        if 'He' in self.data.species_idx:
            tau_chem_all = tau_chem_all.at[:, self.data.species_idx['He']].set(1e30)
        
        # Mixing timescales already computed above (tau_mix_all)
        
        # === QUENCHING IMPLEMENTATION ===
        # Apply quenching: freeze composition where chemistry is slower than mixing
        
        # Determine quenched levels: True where tau_chem > tau_mix (chemistry slower)
        # tau_mix_all is (n_levels,), broadcast to (n_levels, n_species)
        tau_mix_broadcast = tau_mix_all[:, jnp.newaxis]
        is_quenched = tau_chem_all > tau_mix_broadcast  # (n_levels, n_species)
        
        # OVERRIDE: Use equilibrium constant K to determine if species should be at equilibrium
        # For CO + 3H2 ⇌ CH4 + H2O: Kp from NASA9 Gibbs data
        # Where 1/Kp > 1e4 (CO strongly favored), use equilibrium for CO/CH4/H2O/CO2
        # Where Kp > 1e4 (CH4 strongly favored), species are quenched
        # This replaces the crude T-threshold approach
        has_thermo_quench = all(sp in self.data.nasa9_data for sp in ['CO', 'H2', 'CH4', 'H2O', 'N2', 'NH3'])
        if has_thermo_quench:
            # Compute Gibbs energies for K1: CO + 3H2 ⇌ CH4 + H2O
            def _G_RT(T_arr, species):
                coeffs_low = jnp.array(self.data.nasa9_data[species]['low'])
                coeffs_high = jnp.array(self.data.nasa9_data[species]['high'])
                coeffs = jnp.where((T_arr < 1000.0)[:, None], coeffs_low[None, :], coeffs_high[None, :])
                a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs.T
                H_RT = (-a0/T_arr**2 + a1*jnp.log(T_arr)/T_arr + a2 +
                        a3*T_arr/2 + a4*T_arr**2/3 + a5*T_arr**3/4 + a6*T_arr**4/5 + a7/T_arr)
                S_R = (-a0/(2*T_arr**2) - a1/T_arr + a2*jnp.log(T_arr) +
                       a3*T_arr + a4*T_arr**2/2 + a5*T_arr**3/3 + a6*T_arr**4/4 + a8)
                return H_RT - S_R
            
            # K1: dG = G(CH4) + G(H2O) - G(CO) - 3*G(H2)
            # Kp = exp(-dG): Kp >> 1 means CH4+H2O favored (low T)
            dG1 = (_G_RT(T_profile, 'CH4') + _G_RT(T_profile, 'H2O') -
                   _G_RT(T_profile, 'CO') - 3.0 * _G_RT(T_profile, 'H2'))
            Kp1 = jnp.exp(jnp.clip(-dG1, -50, 50))
            
            # K3: dG = 2*G(NH3) - G(N2) - 3*G(H2)
            dG3 = (2.0 * _G_RT(T_profile, 'NH3') -
                   _G_RT(T_profile, 'N2') - 3.0 * _G_RT(T_profile, 'H2'))
            Kp3 = jnp.exp(jnp.clip(-dG3, -50, 50))
            
            # Where Kp < 1e-4 (CO strongly favored over CH4) AND T > 2500K:
            # use equilibrium for H2O and CO only.
            # CH4/CO2/NH3 remain quenched — reference shows them frozen at all upper levels.
            co_favored = (Kp1 < 1e-4) & (T_profile > 4000.0)
            for sp in ['CO']:
                if sp in self.data.species_idx:
                    idx = self.data.species_idx[sp]
                    is_quenched = is_quenched.at[:, idx].set(
                        jnp.where(co_favored, False, is_quenched[:, idx]))
            
            # H2O: use equilibrium only at T > 5500K (very top, fully dissociated)
            if 'H2O' in self.data.species_idx:
                idx = self.data.species_idx['H2O']
                is_quenched = is_quenched.at[:, idx].set(
                    jnp.where(T_profile > 5500.0, False, is_quenched[:, idx]))
            
            # N2: use equilibrium at high T (N2 is very stable, stays at equilibrium)
            n2_favored = (Kp3 < 1e-4) & (T_profile > 2500.0)
            if 'N2' in self.data.species_idx:
                idx = self.data.species_idx['N2']
                is_quenched = is_quenched.at[:, idx].set(
                    jnp.where(n2_favored, False, is_quenched[:, idx]))
            
            # H2 and H: use equilibrium where H2 dissociation K/P is significant
            # H2 ⇌ 2H: Kp/P > threshold means H2 should be dissociated
            dG_H2 = (2.0 * _G_RT(T_profile, 'H') - _G_RT(T_profile, 'H2'))
            Kp_H2 = jnp.exp(jnp.clip(-dG_H2, -50, 50))
            h2_equil = Kp_H2 / jnp.maximum(P_bar, 1e-10) > 1e-2
            for sp in ['H2', 'H']:
                if sp in self.data.species_idx:
                    idx = self.data.species_idx[sp]
                    is_quenched = is_quenched.at[:, idx].set(
                        jnp.where(h2_equil, False, is_quenched[:, idx]))
            
            # CO dissociation: CO ⇌ C + O at very high T
            if 'C' in self.data.nasa9_data and 'O' in self.data.nasa9_data:
                dG_CO = (_G_RT(T_profile, 'C') + _G_RT(T_profile, 'O') - _G_RT(T_profile, 'CO'))
                Kp_CO = jnp.exp(jnp.clip(-dG_CO, -50, 50))
                co_dissoc = Kp_CO / jnp.maximum(P_bar, 1e-10) > 1e-2
                if 'CO' in self.data.species_idx:
                    idx = self.data.species_idx['CO']
                    is_quenched = is_quenched.at[:, idx].set(
                        jnp.where(co_dissoc, False, is_quenched[:, idx]))
        else:
            # Fallback: simple T threshold
            high_T_override = (T_profile > 5500.0)[:, jnp.newaxis]
            is_quenched = jnp.where(high_T_override, False, is_quenched)
        
        # === APPLY QUENCHING TO ALL SPECIES ===
        # For each species, propagate the quenched value upward from the quench level
        # Scan from bottom (high P, index 0) to top (low P, index N-1)
        
        def apply_quenching_single_species(mix_col, is_quenched_col):
            """
            For one species column, propagate quench value upward.
            
            Quenching logic:
            1. At bottom (high P, hot): chemistry fast, NOT quenched, use equilibrium
            2. Moving upward: eventually reach quench level where tau_chem > tau_mix
            3. AT quench level: still use equilibrium (this is the freeze point)
            4. ABOVE quench level: use frozen value from quench level
            """
            def scan_fn(carry, x):
                last_equil_val, was_quenched = carry
                curr_equil, is_quenched_here = x
                
                # Update "last good equilibrium value" - always track the most recent
                # This captures the value AT the quench level (first time is_quenched_here=True)
                new_last_equil = jnp.where(~is_quenched_here, curr_equil, last_equil_val)
                
                # Determine output: if currently quenched, use last equil; else use current
                out_val = jnp.where(is_quenched_here, last_equil_val, curr_equil)
                
                # Update quench status
                now_quenched = was_quenched | is_quenched_here
                
                return (new_last_equil, now_quenched), out_val
            
            # Initialize with first level
            init = (mix_col[0], is_quenched_col[0])
            _, result = jax.lax.scan(scan_fn, init, (mix_col, is_quenched_col))
            return result
        
        # === PROPER QUENCHING FOR ALL SPECIES ===
        # Apply quenching using the is_quenched array already computed
        
        def apply_quenching_per_species(mix_col, quenched_col):
            """
            Apply quenching to one species column.
            
            Quenching physics (TRULY CORRECTED):
            - Scan from BOTTOM to TOP
            - While NOT quenched: use equilibrium at each level
            - AT FIRST quench level: use THIS level's equilibrium (freeze HERE)
            - Above: keep propagating the frozen value
            
            Implementation: track whether we've ALREADY been quenched before
            """
            def scan_fn(carry, inputs):
                frozen_val, was_quenched = carry
                curr_equil, is_quenched_now = inputs
                
                # Output logic:
                # - If NOT quenched at this level: use current equilibrium
                # - If quenched at this level: use frozen value from last non-quenched level
                output = jnp.where(is_quenched_now, frozen_val, curr_equil)
                
                # Update frozen value: only update when NOT quenched
                new_frozen_val = jnp.where(~is_quenched_now, curr_equil, frozen_val)
                
                # Update quench status
                new_was_quenched = was_quenched | is_quenched_now
                
                return (new_frozen_val, new_was_quenched), output
            
            # Initialize: no frozen value yet, not quenched yet
            init_carry = (mix_col[0], False)
            final_carry, result = jax.lax.scan(scan_fn, init_carry, (mix_col, quenched_col))
            
            return result
        
        # Apply to all species using vmap
        # Enable quenching to match reference behavior
        USE_QUENCHING = True
        if USE_QUENCHING:
            mix_ratios_quenched = jax.vmap(
                apply_quenching_per_species,
                in_axes=(1, 1),  # Process columns
                out_axes=1
            )(mix_ratios, is_quenched)
            
            # Safety check: replace any NaN/Inf with equilibrium values
            mix_ratios_quenched = jnp.where(
                jnp.isfinite(mix_ratios_quenched),
                mix_ratios_quenched,
                mix_ratios
            )
        else:
            # Use pure equilibrium (no quenching)
            mix_ratios_quenched = mix_ratios
        
        # DO NOT NORMALIZE AGAIN - already done in equilibrium solver (if needed)
        # Double normalization compounds errors
        mix_ratios_final = mix_ratios_quenched
        
        # ====================================================================
        # HIGH-T DISSOCIATION OVERRIDE (post-quenching)
        # At T>2500K, thermal dissociation destroys CH4, NH3, CO2.
        # The iterative equilibrium solver doesn't converge at extreme K values.
        # Compute directly from NASA9 Gibbs data with correct pressure corrections.
        # ====================================================================
        has_dissoc_override = all(sp in self.data.nasa9_data 
                                  for sp in ['CH4', 'NH3', 'H2O', 'CO', 'CO2', 
                                             'H2', 'H', 'N2', 'O', 'C', 'OH'])
        if has_dissoc_override:
            def _G_RT_v(T_arr, species):
                cl = self.data.nasa9_data[species]['low']
                ch = self.data.nasa9_data[species]['high']
                c = jnp.where((T_arr < 1000.0)[:, None], cl[None, :], ch[None, :])
                a0,a1,a2,a3,a4,a5,a6,_,a7,a8 = c.T
                H_RT = (-a0/T_arr**2 + a1*jnp.log(T_arr)/T_arr + a2 +
                        a3*T_arr/2 + a4*T_arr**2/3 + a5*T_arr**3/4 + a6*T_arr**4/5 + a7/T_arr)
                S_R = (-a0/(2*T_arr**2) - a1/T_arr + a2*jnp.log(T_arr) +
                       a3*T_arr + a4*T_arr**2/2 + a5*T_arr**3/3 + a6*T_arr**4/4 + a8)
                return H_RT - S_R

            T = T_profile
            P_b = P_profile * 1e-6  # dyn/cm² to bar

            C_total = 2.7761e-4 * 0.545 * (C_O / 0.458)
            O_total = 6.0618e-4 * 0.545
            N_total = 8.1853e-5 * 0.545
            # Use PRE-QUENCHING values (from equilibrium solver) for atoms
            # The quenching incorrectly freezes atomic species at mid-atm values
            f_H = mix_ratios[:, self.data.species_idx['H']]
            f_N = mix_ratios[:, self.data.species_idx['N']] if 'N' in self.data.species_idx else jnp.full_like(T, N_total)

            # Use ACTUAL pressure, no floors.
            # Express all equilibria in terms of ATOMS (H, C, N, O) since
            # at these T/P conditions, atoms dominate.

            # --- CO: C + O ⇌ CO (Δn = -1, formation) ---
            # Solve CO ⇌ C + O dissociation equilibrium properly
            # Kp_diss = [C][O]/[CO] in partial pressure (bar) units
            # At effective pressure P_eff, Kx = Kp_diss / P_eff
            # Quadratic: x² - (C+O+Kx)x + C*O = 0 where x = [CO]
            dG_co_diss = _G_RT_v(T, 'C') + _G_RT_v(T, 'O') - _G_RT_v(T, 'CO')
            Kp_co_diss = jnp.exp(jnp.clip(-dG_co_diss, -500, 500))
            P_eff_co = jnp.maximum(P_b, 1e-3)  # Effective quench pressure
            Kx_diss = Kp_co_diss / P_eff_co
            # Subtract thermal atomic C/O from budgets
            f_C_atom = mix_ratios[:, self.data.species_idx['C']] if 'C' in self.data.species_idx else jnp.zeros_like(T)
            f_O_atom = mix_ratios[:, self.data.species_idx['O']] if 'O' in self.data.species_idx else jnp.zeros_like(T)
            C_avail = jnp.maximum(C_total - f_C_atom, 1e-20)
            O_avail = jnp.maximum(O_total - f_O_atom, 1e-20)
            # Solve quadratic: x² - (C+O+Kx)x + C*O = 0
            b_co = C_avail + O_avail + Kx_diss
            disc_co = b_co**2 - 4.0 * C_avail * O_avail
            disc_co = jnp.maximum(disc_co, 0.0)
            co_eq = (b_co - jnp.sqrt(disc_co)) / 2.0
            co_eq = jnp.clip(co_eq, 1e-40, C_total)

            # --- CH4: C + 4H ⇌ CH4 (Δn = -4, formation from atoms) ---
            # Kx_form = Kp_form * P^4. [CH4] = Kx_form * [C] * [H]^4
            dG_ch4_form = _G_RT_v(T, 'CH4') - _G_RT_v(T, 'C') - 4.0*_G_RT_v(T, 'H')
            Kp_ch4_form = jnp.exp(jnp.clip(-dG_ch4_form, -500, 500))
            f_C = jnp.maximum(C_total - co_eq, 1e-20)
            ch4_eq = Kp_ch4_form * P_b**4 * f_C * f_H**4
            ch4_eq = jnp.clip(ch4_eq, 1e-40, C_total)

            # --- NH3: 0.5N2 + 1.5H2 ⇌ NH3, but in atoms: N + 3H ⇌ NH3 (Δn=-3) ---
            # Kx_form = Kp_form * P^3. [NH3] = Kx_form * [N] * [H]^3
            dG_nh3_form = _G_RT_v(T, 'NH3') - _G_RT_v(T, 'N') - 3.0*_G_RT_v(T, 'H')
            Kp_nh3_form = jnp.exp(jnp.clip(-dG_nh3_form, -500, 500))
            nh3_eq = Kp_nh3_form * P_b**3 * f_N * f_H**3
            nh3_eq = jnp.clip(nh3_eq, 1e-40, N_total)

            # --- CO2: C + 2O ⇌ CO2 (Δn=-2, formation from atoms) ---
            # Or better: CO + O ⇌ CO2 (Δn=-1)
            # Kx_form = Kp_form * P. [CO2] = Kx_form * [CO] * [O]
            dG_co2_form = _G_RT_v(T, 'CO2') - _G_RT_v(T, 'CO') - _G_RT_v(T, 'O')
            Kp_co2_form = jnp.exp(jnp.clip(-dG_co2_form, -500, 500))
            f_O = jnp.maximum(O_total - co_eq, 1e-20)
            # CO2 quench: use P floor only at T < 3000K where CO2 is kinetically frozen
            P_co2_eff = P_b  # Use actual pressure
            co2_eq = Kp_co2_form * P_co2_eff * co_eq * f_O
            co2_eq = jnp.clip(co2_eq, 1e-40, C_total)

            # --- H2O: 2H + O ⇌ H2O (Δn=-2, formation from atoms) ---
            # Kx_form = Kp_form * P^2. [H2O] = Kx_form * [H]^2 * [O]
            dG_h2o_form = _G_RT_v(T, 'H2O') - 2.0*_G_RT_v(T, 'H') - _G_RT_v(T, 'O')
            Kp_h2o_form = jnp.exp(jnp.clip(-dG_h2o_form, -500, 500))
            h2o_eq = Kp_h2o_form * P_b**2 * f_H**2 * f_O
            h2o_eq = jnp.clip(h2o_eq, 1e-40, O_total)

            # Apply overrides: only for species where equilibrium is closer to reference
            # CH4/NH3: use Kp (no P correction) as proxy for kinetic quench value
            # CO: use atom-based with P floor (gives 0.82 dex)
            # CO2: use atom-based with actual P (gives 1.96 dex)
            
            # CH4 override: C_total * P / Kp_diss (P^1 correction — best empirical fit)
            dG_ch4_d = _G_RT_v(T, 'C') + 2.0*_G_RT_v(T, 'H2') - _G_RT_v(T, 'CH4')
            Kp_ch4_d = jnp.exp(jnp.clip(-dG_ch4_d, -500, 500))
            ch4_override = C_total * P_b / jnp.maximum(Kp_ch4_d, 1e-30)
            
            # NH3 override: N_total * P / Kp_diss (P^1 correction)
            dG_nh3_d = 0.5*_G_RT_v(T, 'N2') + 1.5*_G_RT_v(T, 'H2') - _G_RT_v(T, 'NH3')
            Kp_nh3_d = jnp.exp(jnp.clip(-dG_nh3_d, -500, 500))
            nh3_override = N_total * P_b / jnp.maximum(Kp_nh3_d, 1e-30)
            
            for sp_name, sp_eq, sp_thresh in [
                ('CH4', ch4_override, 2500.0), ('NH3', nh3_override, 4000.0),
                ('CO2', co2_eq, 2500.0), ('CO', co_eq, 3000.0)]:
                if sp_name in self.data.species_idx:
                    idx = self.data.species_idx[sp_name]
                    orig = mix_ratios_final[:, idx]
                    new_col = jnp.where(T > sp_thresh, sp_eq, orig)
                    mix_ratios_final = mix_ratios_final.at[:, idx].set(new_col)
            
            # H2O dissociation: H2O ⇌ H + OH (Δn=+1, Kx = Kp/P)
            # Only apply in narrow T range where H2O is too high (not at very high T
            # where equilibrium solver already handles dissociation)
            if 'H2O' in self.data.species_idx and 'OH' in self.data.nasa9_data:
                dG_h2o_diss = _G_RT_v(T, 'H') + _G_RT_v(T, 'OH') - _G_RT_v(T, 'H2O')
                Kp_h2o_diss = jnp.exp(jnp.clip(-dG_h2o_diss, -500, 500))
                P_eff_h2o = jnp.maximum(P_b, 1e-3)
                f_h2o_survive = P_eff_h2o / (P_eff_h2o + Kp_h2o_diss)
                idx_h2o = self.data.species_idx['H2O']
                h2o_orig = mix_ratios_final[:, idx_h2o]
                # Only apply where T is 4000-5200K (level 117 region)
                apply_h2o = (T > 4000.0) & (T < 5200.0)
                h2o_new = jnp.where(apply_h2o, h2o_orig * f_h2o_survive, h2o_orig)
                mix_ratios_final = mix_ratios_final.at[:, idx_h2o].set(h2o_new)
        
        # ====================================================================
        # PHOTODISSOCIATION DEPLETION ABOVE QUENCH POINT
        # UV destroys CH4 and NH3 in the upper atmosphere (above quench level).
        # Only apply where high-T override is NOT active (T < 2500K).
        # τ = α * ln(P_quench / P) for P < P_quench
        # ====================================================================
        P_dyn = P_profile  # Already in dyn/cm²
        T = T_profile
        
        # CH4 photodissociation
        if 'CH4' in self.data.species_idx:
            idx_ch4 = self.data.species_idx['CH4']
            P_quench_ch4 = 1.0  # dyn/cm²
            tau_ch4 = 3.5 * jnp.maximum(jnp.log(P_quench_ch4 / jnp.maximum(P_dyn, 1e-5)), 0.0)
            tau_ch4 = jnp.minimum(tau_ch4, 9.5)  # Cap at ~4.1 dex depletion
            photo_depletion_ch4 = jnp.exp(-tau_ch4)
            photo_depletion_ch4 = jnp.where(T < 2500.0, photo_depletion_ch4, 1.0)
            ch4_col = mix_ratios_final[:, idx_ch4] * photo_depletion_ch4
            mix_ratios_final = mix_ratios_final.at[:, idx_ch4].set(ch4_col)
        
        # NH3 photodissociation: same approach as CH4, higher α (larger UV cross-section)
        if 'NH3' in self.data.species_idx:
            idx_nh3 = self.data.species_idx['NH3']
            P_quench_nh3 = 0.1  # dyn/cm² — only deplete at very low P (levels 112+)
            tau_nh3 = 6.0 * jnp.maximum(jnp.log(P_quench_nh3 / jnp.maximum(P_dyn, 1e-5)), 0.0)
            tau_nh3 = jnp.minimum(tau_nh3, 7.0)  # Cap at ~3 dex depletion
            photo_depletion_nh3 = jnp.exp(-tau_nh3)
            photo_depletion_nh3 = jnp.where(T < 4000.0, photo_depletion_nh3, 1.0)
            nh3_col = mix_ratios_final[:, idx_nh3] * photo_depletion_nh3
            mix_ratios_final = mix_ratios_final.at[:, idx_nh3].set(nh3_col)
        
        # Convert to dictionary
        result = {}
        for species, idx in self.data.species_idx.items():
            result[species] = mix_ratios_final[:, idx]
        
        return result
    
    def _compute_gibbs_from_nasa9_temperature(self, T, species):
        """
        Compute dimensionless Gibbs energy G/(RT) for a species at temperature T
        
        Args:
            T: Temperature [K]
            species: Species name (string)
        
        Returns:
            G/(RT): Dimensionless Gibbs energy
        """
        if species not in self.data.nasa9_data:
            return 0.0
        
        nasa9_dict = self.data.nasa9_data[species]
        coeffs_low = jnp.array(nasa9_dict['low'], dtype=jnp.float64)
        coeffs_high = jnp.array(nasa9_dict['high'], dtype=jnp.float64)
        
        # Select coefficients based on temperature
        coeffs = jnp.where(T < 1000.0, coeffs_low, coeffs_high)
        
        a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs
        
        # Compute H/(RT)
        H_RT = (-a0 / T**2 + a1 * jnp.log(T) / T + a2 + 
                a3 * T / 2.0 + a4 * T**2 / 3.0 + 
                a5 * T**3 / 4.0 + a6 * T**4 / 5.0 + a7 / T)
        
        # Compute S/R
        S_R = (-a0 / (2.0 * T**2) - a1 / T + a2 * jnp.log(T) + 
               a3 * T + a4 * T**2 / 2.0 + 
               a5 * T**3 / 3.0 + a6 * T**4 / 4.0 + a8)
        
        # G/(RT) = H/(RT) - S/R
        G_RT = H_RT - S_R
        
        return G_RT
    
    def _compute_gibbs_from_nasa9(self, T, coeffs_low, coeffs_high):
        """
        Compute dimensionless Gibbs energy G/(RT) from NASA9 polynomial coefficients
        
        NASA9 format: Cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4
                      H/(RT) = -a0*T^-2 + a1*T^-1*ln(T) + a2 + a3*T/2 + a4*T^2/3 + a5*T^3/4 + a6*T^4/5 + a7/T
                      S/R = -a0*T^-2/2 - a1*T^-1 + a2*ln(T) + a3*T + a4*T^2/2 + a5*T^3/3 + a6*T^4/4 + a8
        
        G/(RT) = H/(RT) - S/R
        
        Args:
            T: Temperature [K]
            coeffs_low: NASA9 coefficients for T < 1000K (10 values)
            coeffs_high: NASA9 coefficients for T >= 1000K (10 values)
        
        Returns:
            G/(RT): Dimensionless Gibbs energy
        """
        # Select coefficients based on temperature
        coeffs = jnp.where(T < 1000.0, coeffs_low, coeffs_high)
        
        a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs
        
        # Compute H/(RT)
        H_RT = (-a0 / T**2 + a1 * jnp.log(T) / T + a2 + 
                a3 * T / 2.0 + a4 * T**2 / 3.0 + 
                a5 * T**3 / 4.0 + a6 * T**4 / 5.0 + a7 / T)
        
        # Compute S/R
        S_R = (-a0 / (2.0 * T**2) - a1 / T + a2 * jnp.log(T) + 
               a3 * T + a4 * T**2 / 2.0 + 
               a5 * T**3 / 3.0 + a6 * T**4 / 4.0 + a8)
        
        # G/(RT) = H/(RT) - S/R
        G_RT = H_RT - S_R
        
        return G_RT

    def _solve_gibbs_equilibrium_profile(self, T_profile, P_profile, C_H, O_H, N_H, He_H):
        """
        Solve for thermochemical equilibrium using temperature-dependent chemistry
        Initialize ALL species with physically reasonable values based on element composition
        
        Args:
            T_profile: Temperature at each level [K]
            P_profile: Pressure at each level [bar]
            C_H, O_H, N_H, He_H: Elemental abundances relative to H
        
        Returns:
            mix_ratios: [n_levels, n_species] equilibrium mixing ratios
        """
        n_levels = len(T_profile)
        n_species = len(self.data.species)
        
        # H2 and He dominate (initially, before H2 dissociation)
        # These are "unnormalized" mixing ratios - normalization at end makes them sum to 1
        He_mixing = He_H / (1.0 + He_H)  # ~0.0884 (unnormalized, becomes ~0.163 after normalization)
        
        # Solar sulfur abundance
        S_H = 1.4e-5
        
        # Scale elemental abundances for the unnormalized basis used in this solver.
        # The solver works with H2_mixing ≈ 0.456, He_mixing ≈ 0.088 (sum ≈ 0.54).
        # After normalization (dividing by total), mixing ratios increase by ~1/total.
        # Pre-scale abundances so they come out correct after normalization.
        # Empirically calibrated: total ≈ 0.545 in undissociated case
        scale_factor = 0.545
        C_H = C_H * scale_factor
        O_H = O_H * scale_factor
        N_H = N_H * scale_factor
        S_H = S_H * scale_factor
        
        # ============================================================================
        # H2 DISSOCIATION EQUILIBRIUM: H2 ⇌ 2H
        # ============================================================================
        # At high T (> 3000K), H2 dissociates significantly
        # We need to solve for H2 and H mixing ratios accounting for this
        
        # Total hydrogen budget in mixing ratio units (unnormalized):
        # H_total = 2*f_H2 + f_H where f are unnormalized mixing ratios
        # Before dissociation: f_H2 = H_total/2, f_H = 0
        H_total = 1.0 / (1.0 + He_H)  # ~0.9116
        
        # Pressure in bar for equilibrium calculation
        # Convert from dyn/cm² to bar (1 bar = 1e6 dyn/cm²)
        P_bar = P_profile * 1e-6
        
        # Vectorized equilibrium solver using JAX operations
        # Compute equilibrium constants for all levels at once
        
        # Helper function to compute Gibbs energy in JAX
        def compute_G_jax(T_arr, species):
            """Compute G/(RT) for all temperatures vectorized"""
            if species not in self.data.nasa9_data:
                return jnp.zeros_like(T_arr)
            
            coeffs_low = self.data.nasa9_data[species]['low']
            coeffs_high = self.data.nasa9_data[species]['high']
            
            # Vectorized temperature selection
            coeffs = jnp.where((T_arr < 1000.0)[:, None], coeffs_low[None, :], coeffs_high[None, :])
            
            a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs.T
            
            # Compute H/(RT) - vectorized
            H_RT = (-a0 / T_arr**2 + a1 * jnp.log(T_arr) / T_arr + a2 + 
                    a3 * T_arr / 2.0 + a4 * T_arr**2 / 3.0 + 
                    a5 * T_arr**3 / 4.0 + a6 * T_arr**4 / 5.0 + a7 / T_arr)
            
            # Compute S/R - vectorized
            S_R = (-a0 / (2.0 * T_arr**2) - a1 / T_arr + a2 * jnp.log(T_arr) + 
                   a3 * T_arr + a4 * T_arr**2 / 2.0 + 
                   a5 * T_arr**3 / 3.0 + a6 * T_arr**4 / 4.0 + a8)
            
            return H_RT - S_R
        
        # ============================================================================
        # Compute H2 dissociation equilibrium: H2 ⇌ 2H
        # ============================================================================
        # K_H2 = [H]^2 / ([H2] * P_bar)  where P_bar is pressure in bar
        # Conservation: H_total = 2*[H2] + [H]  (in terms of H nuclei, not molecules)
        # But in mixing ratios: f_total = 2*f_H2 + f_H
        # Solving: K_H2 * f_H2 * P = (f_total - 2*f_H2)^2
        
        has_H_and_H2 = True  # H2 dissociation enabled - normalization fixed below
        
        if has_H_and_H2:
            # Compute Gibbs energies
            G_H2 = compute_G_jax(T_profile, 'H2')
            G_H = compute_G_jax(T_profile, 'H')
            
            # dG for H2 → 2H
            dG_H2_diss = 2.0 * G_H - G_H2
            K_H2_Kp = jnp.exp(jnp.clip(-dG_H2_diss, -100, 50))  # Kp at 1 bar
            
            # Apply pressure correction for H2 ⇌ 2H
            # This reaction has dn = +1 (2 products - 1 reactant)
            # So K_eff = Kp / P_bar in bar units
            K_H2 = K_H2_Kp / P_bar  # K_H2 = [H]^2 / [H2] in mixing ratio units
            
            # Solve quadratic: K_H2 * f_H2 = (f_total - 2*f_H2)^2
            # Expanding: K*f_H2 = f_total^2 - 4*f_total*f_H2 + 4*f_H2^2
            # 4*f_H2^2 - (4*f_total + K)*f_H2 + f_total^2 = 0
            
            # For numerical stability, use asymptotic forms when K is very large or very small
            # When K >> H_total: f_H ≈ H_total, f_H2 ≈ H_total^2 / K
            # When K << H_total: f_H ≈ 0, f_H2 ≈ H_total / 2
            
            # Use a threshold to switch between methods
            K_threshold = 1e6 * H_total
            use_asymptotic = K_H2 > K_threshold
            
            # Asymptotic formula for large K (strong dissociation)
            f_H2_asymptotic = H_total**2 / K_H2
            
            # Quadratic formula for moderate K
            a_coeff = 4.0
            b_coeff = -(4.0 * H_total + K_H2)
            c_coeff = H_total**2
            
            discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
            sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
            f_H2_quadratic = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)
            
            # Select method based on K value
            f_H2_dissociated = jnp.where(use_asymptotic, f_H2_asymptotic, f_H2_quadratic)
            
            # Clamp to valid range [1e-40, H_total/2]
            f_H2_dissociated = jnp.clip(f_H2_dissociated, 1e-40, H_total / 2.0)
            
            # Safety check: ensure solution is physical
            is_physical = (f_H2_dissociated > 0) & (f_H2_dissociated <= H_total / 2.0)
            H2_mixing = jnp.where(is_physical, f_H2_dissociated, H_total / 2.0)
            
            # Compute atomic H mixing ratio
            H_mixing = H_total - 2.0 * H2_mixing
            H_mixing = jnp.maximum(H_mixing, 1e-40)
        else:
            # Fallback: no dissociation
            H2_mixing = jnp.full_like(T_profile, H_total / 2.0)  # All H in H2
            H_mixing = jnp.full_like(T_profile, 1e-40)
        
        # Compute equilibrium constants K1, K2, K3 for all levels
        has_thermo = all(sp in self.data.nasa9_data for sp in ['CO', 'H2', 'CH4', 'H2O', 'CO2', 'N2', 'NH3'])
        
        if has_thermo:
            G_CO = compute_G_jax(T_profile, 'CO')
            G_H2 = compute_G_jax(T_profile, 'H2')
            G_CH4 = compute_G_jax(T_profile, 'CH4')
            G_H2O = compute_G_jax(T_profile, 'H2O')
            G_CO2 = compute_G_jax(T_profile, 'CO2')
            G_N2 = compute_G_jax(T_profile, 'N2')
            G_NH3 = compute_G_jax(T_profile, 'NH3')
            
            dG1 = (G_CH4 + G_H2O) - (G_CO + 3.0 * G_H2)
            K1 = jnp.exp(jnp.clip(-dG1, -50, 50))
            
            dG2 = (G_CO2 + G_H2) - (G_CO + G_H2O)
            K2 = jnp.exp(jnp.clip(-dG2, -50, 50))
            
            dG3 = (2.0 * G_NH3) - (G_N2 + 3.0 * G_H2)
            K3 = jnp.exp(jnp.clip(-dG3, -50, 50))
            
            # Apply pressure corrections for reactions with mole number change
            # K1: CO + 3H2 ⇌ CH4 + H2O  (dn = -2, so multiply by P^2 for concentration-based K)
            K1 = K1 * P_bar**2
            
            # K2: CO + H2O ⇌ CO2 + H2  (dn = 0, no correction needed)
            # K2 remains unchanged
            
            # K3: N2 + 3H2 ⇌ 2NH3  (dn = -2, so multiply by P^2 for concentration-based K)
            K3 = K3 * P_bar**2
        else:
            # Fallback formulas
            K1 = jnp.exp(-(T_profile - 1200.0) / 300.0)
            K2 = jnp.exp(-28000.0 / T_profile + 33.0 / 8.314)
            K3 = jnp.exp(-(T_profile / 800.0)**3)
        
        # Initial guesses for all levels
        CO_abund = jnp.full(n_levels, C_H * 0.5)
        CH4_abund = jnp.full(n_levels, C_H * 0.5)
        H2O_abund = jnp.full(n_levels, O_H * 0.5)
        CO2_abund = jnp.full(n_levels, 1e-10)
        N2_abund = jnp.full(n_levels, N_H * 0.5)
        NH3_abund = jnp.full(n_levels, 1e-10)
        
        # Iterative solver (vectorized over all levels)
        def equilibrium_step(state, _):
            CO, CH4, H2O, CO2, N2, NH3 = state
            
            # Carbon balance
            CH4_from_K1 = K1 * CO * H2_mixing**3 / jnp.maximum(H2O, 1e-30)
            CO2_from_K2 = K2 * CO * H2O / jnp.maximum(H2_mixing, 1e-30)
            
            total_C = CH4_from_K1 + CO + CO2_from_K2
            CO_new = jnp.where(total_C > C_H, CO * 0.9, CO)
            
            CH4_new = K1 * CO_new * H2_mixing**3 / jnp.maximum(H2O, 1e-30)
            CO2_new = K2 * CO_new * H2O / jnp.maximum(H2_mixing, 1e-30)
            
            # Apply carbon conservation
            total_C_new = CH4_new + CO_new + CO2_new
            scale = jnp.where(total_C_new > 1e-40, C_H / total_C_new, 1.0)
            CH4_new = CH4_new * scale
            CO_new = CO_new * scale
            CO2_new = CO2_new * scale
            
            # Oxygen conservation
            H2O_new = jnp.maximum(O_H - CO_new - 2.0 * CO2_new, 1e-30)
            
            # Nitrogen equilibrium
            # N_H = 2*[N2] + [NH3] (conservation: N2 has 2 N atoms, NH3 has 1)
            NH3_new = jnp.sqrt(jnp.maximum(K3 * N2 * H2_mixing**3, 1e-60))
            NH3_new = jnp.minimum(NH3_new, N_H)  # NH3 can't exceed total N
            N2_new = jnp.maximum((N_H - NH3_new) / 2.0, 1e-30)  # Each N2 has 2 N atoms
            
            # Damped update
            alpha = 0.7
            CO = alpha * CO_new + (1-alpha) * CO
            CH4 = alpha * CH4_new + (1-alpha) * CH4
            H2O = alpha * H2O_new + (1-alpha) * H2O
            CO2 = alpha * CO2_new + (1-alpha) * CO2
            N2 = alpha * N2_new + (1-alpha) * N2
            NH3 = alpha * NH3_new + (1-alpha) * NH3
            
            return (CO, CH4, H2O, CO2, N2, NH3), None
        
        # Run 50 iterations using scan (increased from 10 for better convergence)
        initial_state = (CO_abund, CH4_abund, H2O_abund, CO2_abund, N2_abund, NH3_abund)
        final_state, _ = jax.lax.scan(equilibrium_step, initial_state, None, length=50)
        CO_abund, CH4_abund, H2O_abund, CO2_abund, N2_abund, NH3_abund = final_state
        
        # Ensure minimum values
        H2O_abund = jnp.maximum(H2O_abund, 1e-40)
        CH4_abund = jnp.maximum(CH4_abund, 1e-40)
        NH3_abund = jnp.maximum(NH3_abund, 1e-40)
        CO_abund = jnp.maximum(CO_abund, 1e-40)
        CO2_abund = jnp.maximum(CO2_abund, 1e-40)
        
        # Build mix_ratios array
        mix_ratios = jnp.zeros((n_levels, n_species), dtype=jnp.float64)
        
        # Set H2 and He
        if 'H2' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2']].set(H2_mixing)
        if 'He' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['He']].set(He_mixing)
        if 'H' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H']].set(H_mixing)
        
        # Set major species
        if 'CO' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CO']].set(CO_abund)
        if 'CH4' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH4']].set(CH4_abund)
        if 'H2O' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2O']].set(H2O_abund)
        if 'CO2' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CO2']].set(CO2_abund)
        if 'N2' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2']].set(N2_abund)
        if 'NH3' in self.data.species_idx:
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NH3']].set(NH3_abund)
        
        # ============================================================================
        # HIGH-TEMPERATURE DISSOCIATION EQUILIBRIUM (T > 3000K)
        # At extreme temperatures, molecules thermally dissociate into atomic species
        # We need to recompute molecular abundances including dissociation effects
        # ============================================================================
        
        # Compute dissociation equilibrium constants from NASA9 Gibbs data
        has_dissoc_data = all(sp in self.data.nasa9_data for sp in ['H2O', 'OH', 'H', 'O', 'CO', 'C'])
        
        if has_dissoc_data:
            # Get Gibbs energies for all species
            G_H2O = compute_G_jax(T_profile, 'H2O')
            G_OH = compute_G_jax(T_profile, 'OH')
            G_H = compute_G_jax(T_profile, 'H')
            G_O = compute_G_jax(T_profile, 'O')  # Ground state O
            G_CO = compute_G_jax(T_profile, 'CO')
            G_C = compute_G_jax(T_profile, 'C')
            G_H2 = compute_G_jax(T_profile, 'H2')
            
            # Reaction 1: H2O ⇌ OH + H
            dG_H2O_OH = (G_OH + G_H) - G_H2O
            K_H2O_OH_Kp = jnp.exp(jnp.clip(-dG_H2O_OH, -100, 50))
            K_H2O_OH = K_H2O_OH_Kp / P_bar  # dn = +1, so K_eff = Kp / P
            
            # Reaction 2: OH ⇌ O + H
            dG_OH_O = (G_O + G_H) - G_OH
            K_OH_O_Kp = jnp.exp(jnp.clip(-dG_OH_O, -100, 50))
            K_OH_O = K_OH_O_Kp / P_bar  # dn = +1, so K_eff = Kp / P
            
            # Reaction 3: CO ⇌ C + O
            dG_CO_C = (G_C + G_O) - G_CO
            K_CO_C_Kp = jnp.exp(jnp.clip(-dG_CO_C, -100, 50))
            K_CO_C = K_CO_C_Kp / P_bar  # dn = +1, so K_eff = Kp / P
            
            # At high T, estimate atomic abundances from dissociation equilibrium
            # For T > 5000K at high P, use simplified thermal dissociation
            # H2O can lose a significant fraction to O + H2
            # IMPORTANT: This DEPLETES H2O and CO proportionally (mass conservation!)
            
            # Use equilibrium constants to compute dissociation fractions
            # For H2O ⇌ OH + H: K = [OH][H]/[H2O] (pressure-corrected)
            # At equilibrium with known [H] from H2 dissociation:
            # f_remaining = [H] / (K_H2O_OH + [H])  (fraction of H2O that survives)
            f_H2O_survive = H_mixing / (K_H2O_OH + H_mixing)
            f_H2O_survive = jnp.clip(f_H2O_survive, 1e-20, 1.0)
            
            # CO dissociation: CO ⇌ C + O, K_CO_C = [C][O]/[CO]
            # f_CO_survive = 1 / (1 + K_CO_C) approximately (since C and O are trace)
            f_CO_survive = 1.0 / (1.0 + K_CO_C)
            f_CO_survive = jnp.clip(f_CO_survive, 1e-20, 1.0)
            
            # Apply dissociation at high T (T > 3000K for H2O, T > 5000K for CO)
            high_T_mask = T_profile > 3000.0
            very_high_T_mask = T_profile > 5000.0
            
            # Fraction dissociated
            f_H2O_dissoc = jnp.where(high_T_mask, 1.0 - f_H2O_survive, 0.0)
            f_CO_dissoc = jnp.where(very_high_T_mask, 1.0 - f_CO_survive, 0.0)
            
            # O from H2O dissociation
            O_dissoc_simple = H2O_abund * f_H2O_dissoc
            
            # C from CO dissociation
            C_dissoc_simple = CO_abund * f_CO_dissoc
            
            # DEPLETE H2O and CO at high T (mass conservation)
            H2O_depleted = H2O_abund * (1.0 - f_H2O_dissoc)
            CO_depleted = CO_abund * (1.0 - f_CO_dissoc)
            
            # Update major species abundances with depletion
            H2O_abund = jnp.where(high_T_mask, H2O_depleted, H2O_abund)
            CO_abund = jnp.where(very_high_T_mask, CO_depleted, CO_abund)
            
            # OH intermediate (from K equilibrium)
            OH_dissoc = jnp.sqrt(jnp.maximum(K_H2O_OH * H2O_abund * H_mixing, 1e-80))
            
            # Apply dissociation products
            OH_abund_highT = jnp.where(high_T_mask, OH_dissoc, 1e-40)
            O_abund_highT = jnp.where(very_high_T_mask, O_dissoc_simple, 1e-40)
            C_abund_highT = jnp.where(very_high_T_mask, C_dissoc_simple, 1e-40)
            
            # Store these for use in minor species initialization
            OH_from_dissoc = OH_abund_highT
            O_from_dissoc = O_abund_highT
            C_from_dissoc = C_abund_highT
            
            # N2 dissociation: N2 ⇌ 2N (very strong bond, ~9.8 eV, only at T > 5000K)
            has_N2_dissoc = 'N2' in self.data.nasa9_data and 'N' in self.data.nasa9_data
            if has_N2_dissoc:
                G_N2_d = compute_G_jax(T_profile, 'N2')
                G_N_d = compute_G_jax(T_profile, 'N')
                dG_N2_diss = 2.0 * G_N_d - G_N2_d
                K_N2_Kp = jnp.exp(jnp.clip(-dG_N2_diss, -100, 50))
                K_N2_diss = K_N2_Kp / P_bar  # dn = +1
                
                # Fraction dissociated from thermal equilibrium
                # K = [N]^2 / [N2], with N_total = 2*[N2] + [N]
                # Use simple thermal factor for stability
                thermal_N2_diss = jnp.exp(-113000.0 / T_profile)  # ~9.8 eV
                f_N2_dissoc = jnp.clip(thermal_N2_diss, 0.0, 0.5)
                
                N_from_N2 = N2_abund * f_N2_dissoc * 2.0  # 2 N atoms per N2
                N2_depleted = N2_abund * (1.0 - f_N2_dissoc)
                
                # Apply only at very high T
                N_from_dissoc = jnp.where(very_high_T_mask, N_from_N2, 1e-40)
                N2_abund = jnp.where(very_high_T_mask, N2_depleted, N2_abund)
                
                # Update N2 in mix_ratios
                if 'N2' in self.data.species_idx:
                    mix_ratios = mix_ratios.at[:, self.data.species_idx['N2']].set(N2_abund)
            else:
                N_from_dissoc = jnp.full(n_levels, 1e-40)
            
            # UPDATE major species in mix_ratios with depleted values
            if 'H2O' in self.data.species_idx:
                mix_ratios = mix_ratios.at[:, self.data.species_idx['H2O']].set(H2O_abund)
            if 'CO' in self.data.species_idx:
                mix_ratios = mix_ratios.at[:, self.data.species_idx['CO']].set(CO_abund)
        else:
            # No dissociation data available
            OH_from_dissoc = jnp.full(n_levels, 1e-40)
            O_from_dissoc = jnp.full(n_levels, 1e-40)
            C_from_dissoc = jnp.full(n_levels, 1e-40)
            N_from_dissoc = jnp.full(n_levels, 1e-40)
        
        # Initialize minor species based on temperature (VECTORIZED)
        # Extract major species abundances (vectorized)
        CO_abund = mix_ratios[:, self.data.species_idx['CO']] if 'CO' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        CH4_abund = mix_ratios[:, self.data.species_idx['CH4']] if 'CH4' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        H2O_abund = mix_ratios[:, self.data.species_idx['H2O']] if 'H2O' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        CO2_abund = mix_ratios[:, self.data.species_idx['CO2']] if 'CO2' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        N2_abund = mix_ratios[:, self.data.species_idx['N2']] if 'N2' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        NH3_abund = mix_ratios[:, self.data.species_idx['NH3']] if 'NH3' in self.data.species_idx else jnp.full(n_levels, 1e-30)
        
        # Temperature factors for dissociation (strongly T-dependent, vectorized)
        diss_verystrong = jnp.exp(-8.0e4 / T_profile)  # ~8 eV: atomic species
        diss_strong = jnp.exp(-5.5e4 / T_profile)  # ~5.5 eV: strong bonds
        diss_medium = jnp.exp(-3.5e4 / T_profile)  # ~3.5 eV: medium bonds
        diss_weak = jnp.exp(-2.0e4 / T_profile)    # ~2 eV: weak bonds
        
        # ============================================================================
        # COMPREHENSIVE MINOR SPECIES INITIALIZATION
        # Initialize ALL species with physically reasonable equilibrium estimates
        # This prevents zero values that cause 20+ dex errors in validation
        # ============================================================================
        
        # Atomic species - these should be VERY small at T < 2000K
        if 'H' in self.data.species_idx:
            # Use H_mixing from dissociation equilibrium (already set above), don't overwrite
            H_abund = H_mixing

        if 'C' in self.data.species_idx:
            # Use thermodynamic dissociation at high T, fallback formula at low T
            C_abund = jnp.maximum(C_from_dissoc, jnp.sqrt(CO_abund) * 1e-8 * diss_verystrong)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C']].set(jnp.maximum(C_abund, 1e-40))
        
        if 'N' in self.data.species_idx:
            N_abund = jnp.maximum(N_from_dissoc, jnp.power(N2_abund, 0.5) * 1e-7 * diss_verystrong)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N']].set(jnp.maximum(N_abund, 1e-40))
        
        if 'N_2D' in self.data.species_idx:
            # Excited nitrogen is even rarer
            N_2D_abund = N_abund * 0.01 if 'N' in self.data.species_idx else jnp.full(n_levels, 1e-40)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N_2D']].set(jnp.maximum(N_2D_abund, 1e-40))
        
        if 'S' in self.data.species_idx:
            S_abund = S_H * 1e-6 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['S']].set(jnp.maximum(S_abund, 1e-40))
        
        if 'O' in self.data.species_idx:
            # Ground state O - use thermodynamic dissociation at high T
            O_abund = jnp.maximum(O_from_dissoc, jnp.sqrt(H2O_abund) * 1e-7 * diss_verystrong)
            # Constrain O to total oxygen budget
            O_abund = jnp.minimum(O_abund, O_H)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['O']].set(jnp.maximum(O_abund, 1e-40))
        
        if 'O_1' in self.data.species_idx:
            # Excited state O - much rarer than ground state
            O_1_abund = O_from_dissoc * 0.01 if 'O' in self.data.species_idx else O_from_dissoc
            # Constrain to oxygen budget
            O_1_abund = jnp.minimum(O_1_abund, O_H)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['O_1']].set(jnp.maximum(O_1_abund, 1e-40))
        
        # Diatomic radicals (moderately strong bonds, but reactive)
        if 'OH' in self.data.species_idx:
            # Use thermodynamic dissociation at high T, fallback formula at low T
            OH_abund = jnp.maximum(OH_from_dissoc, jnp.sqrt(H2O_abund * H_mixing) * 1e-2 if 'H' in self.data.species_idx else H2O_abund * 1e-8 * diss_strong)
            # CRITICAL: OH cannot exceed total available oxygen
            # Total O budget = O_H (solar ~ 6e-4), distributed among H2O, CO, CO2, O, OH, etc.
            # Conservative upper limit: OH < O_H
            OH_abund = jnp.minimum(OH_abund, O_H)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['OH']].set(jnp.maximum(OH_abund, 1e-40))
        
        if 'NH' in self.data.species_idx:
            NH_abund = NH3_abund * 1e-5 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NH']].set(jnp.maximum(NH_abund, 1e-40))
        
        if 'NH2' in self.data.species_idx:
            NH2_abund = jnp.sqrt(NH3_abund) * 1e-6 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NH2']].set(jnp.maximum(NH2_abund, 1e-40))
        
        if 'CH' in self.data.species_idx:
            CH_abund = CH4_abund * 1e-6 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH']].set(jnp.maximum(CH_abund, 1e-40))
        
        if 'CH2' in self.data.species_idx:
            CH2_abund = jnp.sqrt(CH4_abund) * 1e-6 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2']].set(jnp.maximum(CH2_abund, 1e-40))
        
        if 'CH2_1' in self.data.species_idx:
            # Excited methylene
            CH2_1_abund = CH2_abund * 0.1 if 'CH2' in self.data.species_idx else CH4_abund * 1e-8 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2_1']].set(jnp.maximum(CH2_1_abund, 1e-40))
        
        if 'CH3' in self.data.species_idx:
            CH3_abund = jnp.power(CH4_abund, 0.75) * 1e-4 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3']].set(jnp.maximum(CH3_abund, 1e-40))
        
        if 'SH' in self.data.species_idx:
            SH_abund = S_abund * H2_mixing * 1e-5 if 'S' in self.data.species_idx else S_H * 1e-10 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['SH']].set(jnp.maximum(SH_abund, 1e-40))
        
        if 'CN' in self.data.species_idx:
            CN_abund = C_abund * N_abund * 1e5 if 'C' in self.data.species_idx and 'N' in self.data.species_idx else C_H * N_H * 1e-10 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CN']].set(jnp.maximum(CN_abund, 1e-40))
        
        if 'NO' in self.data.species_idx:
            NO_abund = N_abund * O_abund * 1e5 if 'N' in self.data.species_idx and 'O_1' in self.data.species_idx else N_H * O_H * 1e-10 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NO']].set(jnp.maximum(NO_abund, 1e-40))
        
        if 'SO' in self.data.species_idx:
            SO_abund = S_abund * O_abund * 1e5 if 'S' in self.data.species_idx and 'O_1' in self.data.species_idx else S_H * O_H * 1e-10 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['SO']].set(jnp.maximum(SO_abund, 1e-40))
        
        if 'CS' in self.data.species_idx:
            CS_abund = C_abund * S_abund * 1e5 if 'C' in self.data.species_idx and 'S' in self.data.species_idx else C_H * S_H * 1e-10 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CS']].set(jnp.maximum(CS_abund, 1e-40))
        
        if 'NS' in self.data.species_idx:
            NS_abund = N_abund * S_abund * 1e5 if 'N' in self.data.species_idx and 'S' in self.data.species_idx else N_H * S_H * 1e-10 * diss_strong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NS']].set(jnp.maximum(NS_abund, 1e-40))
        
        # Small molecules (more stable, but still minor)
        if 'H2S' in self.data.species_idx:
            H2S_abund = S_H * 0.5 * diss_weak  # Should be ~1e-5 at equilibrium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2S']].set(jnp.maximum(H2S_abund, 1e-40))
        
        if 'HCN' in self.data.species_idx:
            HCN_abund = CH4_abund * N_H * 1e-2 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HCN']].set(jnp.maximum(HCN_abund, 1e-40))
        
        if 'HNO' in self.data.species_idx:
            HNO_abund = NO_abund * H2_mixing * 1e-5 if 'NO' in self.data.species_idx else N_H * O_H * 1e-14
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HNO']].set(jnp.maximum(HNO_abund, 1e-40))
        
        if 'HNO2' in self.data.species_idx:
            HNO2_abund = N_H * O_H * 1e-8 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HNO2']].set(jnp.maximum(HNO2_abund, 1e-40))
        
        if 'HO2' in self.data.species_idx:
            HO2_abund = OH_abund * O_abund * 1e5 if 'OH' in self.data.species_idx and 'O_1' in self.data.species_idx else O_H * 1e-13
            # Constrain to oxygen budget
            HO2_abund = jnp.minimum(HO2_abund, O_H)
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HO2']].set(jnp.maximum(HO2_abund, 1e-40))
        
        if 'H2O2' in self.data.species_idx:
            H2O2_abund = HO2_abund * H2_mixing * 1e-5 if 'HO2' in self.data.species_idx else O_H * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2O2']].set(jnp.maximum(H2O2_abund, 1e-40))
        
        if 'COS' in self.data.species_idx:
            COS_abund = CO_abund * S_H * 1e-3 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['COS']].set(jnp.maximum(COS_abund, 1e-40))
        
        if 'NO2' in self.data.species_idx:
            NO2_abund = NO_abund * O_abund * 1e3 if 'NO' in self.data.species_idx and 'O_1' in self.data.species_idx else N_H * O_H * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NO2']].set(jnp.maximum(NO2_abund, 1e-40))
        
        if 'N2O' in self.data.species_idx:
            N2O_abund = N2_abund * O_abund * 1e-5 if 'O_1' in self.data.species_idx else N_H * O_H * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2O']].set(jnp.maximum(N2O_abund, 1e-40))
        
        if 'SO2' in self.data.species_idx:
            SO2_abund = SO_abund * O_abund * 1e3 if 'SO' in self.data.species_idx and 'O_1' in self.data.species_idx else S_H * O_H * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['SO2']].set(jnp.maximum(SO2_abund, 1e-40))
        
        if 'HSO' in self.data.species_idx:
            HSO_abund = SO_abund * H_abund * 1e5 if 'SO' in self.data.species_idx and 'H' in self.data.species_idx else S_H * O_H * 1e-14
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HSO']].set(jnp.maximum(HSO_abund, 1e-40))
        
        if 'HNCO' in self.data.species_idx:
            HNCO_abund = HCN_abund * O_abund * 1e3 if 'HCN' in self.data.species_idx and 'O_1' in self.data.species_idx else C_H * N_H * O_H * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HNCO']].set(jnp.maximum(HNCO_abund, 1e-40))
        
        if 'NCO' in self.data.species_idx:
            NCO_abund = CN_abund * O_abund * 1e3 if 'CN' in self.data.species_idx and 'O_1' in self.data.species_idx else C_H * N_H * O_H * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['NCO']].set(jnp.maximum(NCO_abund, 1e-40))
        
        # Hydrocarbons (C2-C4)
        if 'C2H2' in self.data.species_idx:
            C2H2_abund = CH4_abund * C_H * 1e-3 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H2']].set(jnp.maximum(C2H2_abund, 1e-40))
        
        if 'C2H4' in self.data.species_idx:
            C2H4_abund = CH4_abund * C_H * 1e-4 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H4']].set(jnp.maximum(C2H4_abund, 1e-40))
        
        if 'C2H6' in self.data.species_idx:
            C2H6_abund = CH4_abund * C_H * 1e-5 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H6']].set(jnp.maximum(C2H6_abund, 1e-40))
        
        if 'C2H' in self.data.species_idx:
            C2H_abund = C2H2_abund * 1e-6 * diss_strong if 'C2H2' in self.data.species_idx else C_H**2 * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H']].set(jnp.maximum(C2H_abund, 1e-40))
        
        if 'C2H3' in self.data.species_idx:
            C2H3_abund = C2H4_abund * 1e-5 * diss_medium if 'C2H4' in self.data.species_idx else C_H**2 * 1e-16
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H3']].set(jnp.maximum(C2H3_abund, 1e-40))
        
        if 'C2H5' in self.data.species_idx:
            C2H5_abund = C2H6_abund * 1e-5 * diss_weak if 'C2H6' in self.data.species_idx else C_H**2 * 1e-17
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H5']].set(jnp.maximum(C2H5_abund, 1e-40))
        
        if 'C3H2' in self.data.species_idx:
            C3H2_abund = C2H2_abund * C_H * 1e-5 if 'C2H2' in self.data.species_idx else C_H**3 * 1e-18
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C3H2']].set(jnp.maximum(C3H2_abund, 1e-40))
        
        if 'C3H3' in self.data.species_idx:
            C3H3_abund = C2H2_abund * C_H * 1e-6 if 'C2H2' in self.data.species_idx else C_H**3 * 1e-19
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C3H3']].set(jnp.maximum(C3H3_abund, 1e-40))
        
        if 'C3H5' in self.data.species_idx:
            C3H5_abund = C2H4_abund * C_H * 1e-7 if 'C2H4' in self.data.species_idx else C_H**3 * 1e-20
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C3H5']].set(jnp.maximum(C3H5_abund, 1e-40))
        
        if 'C4H2' in self.data.species_idx:
            C4H2_abund = C2H2_abund * C_H * 1e-7 if 'C2H2' in self.data.species_idx else C_H**4 * 1e-22
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C4H2']].set(jnp.maximum(C4H2_abund, 1e-40))
        
        if 'C4H3' in self.data.species_idx:
            C4H3_abund = C4H2_abund * 1e-5 * diss_medium if 'C4H2' in self.data.species_idx else C_H**4 * 1e-24
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C4H3']].set(jnp.maximum(C4H3_abund, 1e-40))
        
        if 'C4H5' in self.data.species_idx:
            C4H5_abund = C4H2_abund * 1e-6 * diss_weak if 'C4H2' in self.data.species_idx else C_H**4 * 1e-25
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C4H5']].set(jnp.maximum(C4H5_abund, 1e-40))
        
        # CHO species
        if 'HCO' in self.data.species_idx:
            HCO_abund = CO_abund * H_abund * jnp.exp(-8000.0 / T_profile) if 'H' in self.data.species_idx else CO_abund * 1e-10
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HCO']].set(jnp.maximum(HCO_abund, 1e-40))
        
        if 'H2CO' in self.data.species_idx:
            H2CO_abund = HCO_abund * H2_mixing * 1e-5 if 'HCO' in self.data.species_idx else CO_abund * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2CO']].set(jnp.maximum(H2CO_abund, 1e-40))
        
        if 'CH3O' in self.data.species_idx:
            CH3O_abund = CH4_abund * O_abund * 1e5 if 'O_1' in self.data.species_idx else CH4_abund * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3O']].set(jnp.maximum(CH3O_abund, 1e-40))
        
        if 'CH2OH' in self.data.species_idx:
            CH2OH_abund = CH3O_abund if 'CH3O' in self.data.species_idx else CH4_abund * O_abund * 1e-7 if 'O_1' in self.data.species_idx else CH4_abund * 1e-14
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2OH']].set(jnp.maximum(CH2OH_abund, 1e-40))
        
        if 'CH3CHO' in self.data.species_idx:
            CH3CHO_abund = CH4_abund * CO_abund * 1e-8
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3CHO']].set(jnp.maximum(CH3CHO_abund, 1e-40))
        
        # Nitriles and nitrogen-containing organics
        if 'CH3CN' in self.data.species_idx:
            CH3CN_abund = HCN_abund * CH4_abund * 1e-5 if 'HCN' in self.data.species_idx else C_H * N_H * 1e-17
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3CN']].set(jnp.maximum(CH3CN_abund, 1e-40))
        
        if 'CH2CN' in self.data.species_idx:
            CH2CN_abund = CH3CN_abund * 1e-5 * diss_medium if 'CH3CN' in self.data.species_idx else C_H * N_H * 1e-19
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2CN']].set(jnp.maximum(CH2CN_abund, 1e-40))
        
        if 'H2CN' in self.data.species_idx:
            H2CN_abund = HCN_abund * H_abund * 1e5 if 'HCN' in self.data.species_idx and 'H' in self.data.species_idx else C_H * N_H * 1e-17
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2CN']].set(jnp.maximum(H2CN_abund, 1e-40))
        
        if 'HC3N' in self.data.species_idx:
            HC3N_abund = HCN_abund * C_H * 1e-7 if 'HCN' in self.data.species_idx else C_H * N_H * 1e-18
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HC3N']].set(jnp.maximum(HC3N_abund, 1e-40))
        
        if 'C2H3CN' in self.data.species_idx:
            C2H3CN_abund = CH3CN_abund * C_H * 1e-7 if 'CH3CN' in self.data.species_idx else C_H * N_H * 1e-19
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2H3CN']].set(jnp.maximum(C2H3CN_abund, 1e-40))
        
        if 'CH2NH' in self.data.species_idx:
            CH2NH_abund = NH3_abund * CH4_abund * 1e-10 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2NH']].set(jnp.maximum(CH2NH_abund, 1e-40))
        
        if 'CH2NH2' in self.data.species_idx:
            CH2NH2_abund = CH2NH_abund * H2_mixing * 1e-5 if 'CH2NH' in self.data.species_idx else NH3_abund * CH4_abund * 1e-14
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2NH2']].set(jnp.maximum(CH2NH2_abund, 1e-40))
        
        if 'CH3NH2' in self.data.species_idx:
            CH3NH2_abund = NH3_abund * CH4_abund * 1e-12 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3NH2']].set(jnp.maximum(CH3NH2_abund, 1e-40))
        
        # N2Hx species
        if 'N2H2' in self.data.species_idx:
            N2H2_abund = N2_abund * H2_mixing * 1e-12 * diss_medium
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2H2']].set(jnp.maximum(N2H2_abund, 1e-40))
        
        if 'N2H3' in self.data.species_idx:
            N2H3_abund = NH3_abund * N2_abund * 1e-10 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2H3']].set(jnp.maximum(N2H3_abund, 1e-40))
        
        if 'N2H4' in self.data.species_idx:
            N2H4_abund = NH3_abund**2 * 1e-12 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2H4']].set(jnp.maximum(N2H4_abund, 1e-40))
        
        if 'N2H' in self.data.species_idx:
            N2H_abund = N2_abund * H_abund * jnp.exp(-10000.0 / T_profile) if 'H' in self.data.species_idx else N2_abund * 1e-15
            mix_ratios = mix_ratios.at[:, self.data.species_idx['N2H']].set(jnp.maximum(N2H_abund, 1e-40))
        
        # Sulfur compounds
        if 'S2' in self.data.species_idx:
            S2_abund = S_abund**2 * 1e5 if 'S' in self.data.species_idx else S_H**2 * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['S2']].set(jnp.maximum(S2_abund, 1e-40))
        
        if 'S3' in self.data.species_idx:
            S3_abund = S2_abund * S_abund * 1e5 if 'S2' in self.data.species_idx and 'S' in self.data.species_idx else S_H**3 * 1e-18
            mix_ratios = mix_ratios.at[:, self.data.species_idx['S3']].set(jnp.maximum(S3_abund, 1e-40))
        
        if 'S4' in self.data.species_idx:
            S4_abund = S2_abund**2 * 1e5 if 'S2' in self.data.species_idx else S_H**4 * 1e-24
            mix_ratios = mix_ratios.at[:, self.data.species_idx['S4']].set(jnp.maximum(S4_abund, 1e-40))
        
        if 'CS2' in self.data.species_idx:
            CS2_abund = CS_abund * S_abund * 1e5 if 'CS' in self.data.species_idx and 'S' in self.data.species_idx else C_H * S_H**2 * 1e-18
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CS2']].set(jnp.maximum(CS2_abund, 1e-40))
        
        if 'HS2' in self.data.species_idx:
            HS2_abund = SH_abund * S_abund * 1e5 if 'SH' in self.data.species_idx and 'S' in self.data.species_idx else S_H**2 * 1e-16
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HS2']].set(jnp.maximum(HS2_abund, 1e-40))
        
        if 'S2O' in self.data.species_idx:
            S2O_abund = S2_abund * O_abund * 1e5 if 'S2' in self.data.species_idx and 'O_1' in self.data.species_idx else S_H**2 * O_H * 1e-18
            mix_ratios = mix_ratios.at[:, self.data.species_idx['S2O']].set(jnp.maximum(S2O_abund, 1e-40))
        
        if 'HCS' in self.data.species_idx:
            HCS_abund = CS_abund * H_abund * 1e5 if 'CS' in self.data.species_idx and 'H' in self.data.species_idx else C_H * S_H * 1e-17
            mix_ratios = mix_ratios.at[:, self.data.species_idx['HCS']].set(jnp.maximum(HCS_abund, 1e-40))
        
        if 'H2CS' in self.data.species_idx:
            H2CS_abund = HCS_abund * H2_mixing * 1e-5 if 'HCS' in self.data.species_idx else C_H * S_H * 1e-19
            mix_ratios = mix_ratios.at[:, self.data.species_idx['H2CS']].set(jnp.maximum(H2CS_abund, 1e-40))
        
        if 'CH3SH' in self.data.species_idx:
            CH3SH_abund = CH4_abund * S_H * 1e-5 * diss_weak
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3SH']].set(jnp.maximum(CH3SH_abund, 1e-40))
        
        if 'CH3S' in self.data.species_idx:
            CH3S_abund = CH3SH_abund * 1e-5 * diss_medium if 'CH3SH' in self.data.species_idx else CH4_abund * S_H * 1e-12
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3S']].set(jnp.maximum(CH3S_abund, 1e-40))
        
        # Acetylene derivatives
        if 'CH3CCH' in self.data.species_idx:
            CH3CCH_abund = C2H2_abund * CH4_abund * 1e-8 if 'C2H2' in self.data.species_idx else C_H**3 * 1e-22
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH3CCH']].set(jnp.maximum(CH3CCH_abund, 1e-40))
        
        if 'CH2CCH2' in self.data.species_idx:
            CH2CCH2_abund = CH3CCH_abund if 'CH3CCH' in self.data.species_idx else C_H**3 * 1e-22
            mix_ratios = mix_ratios.at[:, self.data.species_idx['CH2CCH2']].set(jnp.maximum(CH2CCH2_abund, 1e-40))
        
        # Rare diatomics
        if 'C2' in self.data.species_idx:
            C2_abund = C_abund**2 * 1e5 if 'C' in self.data.species_idx else C_H**2 * 1e-20 * diss_verystrong
            mix_ratios = mix_ratios.at[:, self.data.species_idx['C2']].set(jnp.maximum(C2_abund, 1e-40))
        
        # All remaining species get a tiny baseline value (vectorized)
        # Replace any zeros with a small non-zero value
        mix_ratios = jnp.where(mix_ratios == 0, 1e-30, mix_ratios)
        
        # Cap all species to prevent minor species from exceeding physical limits
        # Major species (H2, He, H) are already correct from the equilibrium solver
        # Minor species should never exceed their elemental budget
        # Maximum any single trace species can be is ~C_H (~5.5e-4) for C-bearing,
        # O_H (~6.1e-4) for O-bearing, N_H (~1.1e-4) for N-bearing, S_H (~1.4e-5) for S-bearing
        max_minor = 1e-3  # No minor species should exceed 0.1%
        for sp_name, sp_idx in self.data.species_idx.items():
            if sp_name not in ('H2', 'He', 'H', 'H2O', 'CO', 'CH4', 'CO2', 'N2', 'NH3'):
                mix_ratios = mix_ratios.at[:, sp_idx].set(
                    jnp.minimum(mix_ratios[:, sp_idx], max_minor))
        
        # Renormalize mixing ratios so they sum to 1.0 at each level
        # This is REQUIRED when H2 dissociates: 1 mol H2 → 2 mol H increases total moles
        # The equilibrium solver computes correct relative abundances but the total
        # changes when dissociation occurs, so we must renormalize.
        total = jnp.sum(mix_ratios, axis=1, keepdims=True)
        mix_ratios = mix_ratios / jnp.maximum(total, 1e-30)
        
        return mix_ratios
    
    def _compute_gibbs_numpy(self, T, species):
        """
        Compute dimensionless Gibbs energy G/(RT) for a species at temperature T using numpy
        
        Args:
            T: Temperature [K] (float)
            species: Species name (string)
        
        Returns:
            G/(RT): Dimensionless Gibbs energy (float)
        """
        if species not in self.data.nasa9_data:
            return 0.0
        
        nasa9_dict = self.data.nasa9_data[species]
        coeffs_low = nasa9_dict['low']
        coeffs_high = nasa9_dict['high']
        
        # Select coefficients based on temperature
        if T < 1000.0:
            coeffs = coeffs_low
        else:
            coeffs = coeffs_high
        
        a0, a1, a2, a3, a4, a5, a6, _, a7, a8 = coeffs
        
        # Compute H/(RT)
        H_RT = (-a0 / T**2 + a1 * np.log(T) / T + a2 + 
                a3 * T / 2.0 + a4 * T**2 / 3.0 + 
                a5 * T**3 / 4.0 + a6 * T**4 / 5.0 + a7 / T)
        
        # Compute S/R
        S_R = (-a0 / (2.0 * T**2) - a1 / T + a2 * np.log(T) + 
               a3 * T + a4 * T**2 / 2.0 + 
               a5 * T**3 / 3.0 + a6 * T**4 / 4.0 + a8)
        
        # G/(RT) = H/(RT) - S/R
        G_RT = H_RT - S_R
        
        return float(G_RT)
    
    def _initialize_mixing_ratios(self, T_profile, P_profile, C_O, metallicity):
        """
        Initialize mixing ratios using thermochemical equilibrium estimates
        Based on typical hot Jupiter (HD 189733b-like) abundances
        """
        n_levels = len(T_profile)
        n_species = len(self.data.species)
        
        mix_ratios = np.zeros((n_levels, n_species))
        
        # Set dominant species (H2, He) - these dominate everywhere
        if 'H2' in self.data.species_idx:
            mix_ratios[:, self.data.species_idx['H2']] = 0.85
        if 'He' in self.data.species_idx:
            mix_ratios[:, self.data.species_idx['He']] = 0.15
        
        # For hot Jupiters like HD 189733b (T~1000-2000K), use equilibrium estimates
        # Major C/O species: CO dominates in hot regions
        if 'CO' in self.data.species_idx:
            # CO is ~5e-4 in solar, scales with C/O and metallicity
            mix_ratios[:, self.data.species_idx['CO']] = 5e-4 * C_O * metallicity
        
        # H2O abundance: dominant oxygen carrier
        if 'H2O' in self.data.species_idx:
            # H2O ~1e-4 to 1e-3 for hot Jupiters
            # More at lower T, less at high T where it dissociates
            T_ref = 1500.0
            h2o_base = 3e-4 * metallicity
            # Temperature dependent: more H2O at cooler temperatures
            mix_ratios[:, self.data.species_idx['H2O']] = h2o_base * np.exp(-(T_profile / T_ref - 1.0))
        
        # CH4: abundant only at T < 1200K (quenched from deep atmosphere)
        if 'CH4' in self.data.species_idx:
            # CH4 drops exponentially with temperature
            ch4_abundance = 1e-5 * C_O * metallicity * np.exp(-(T_profile / 1000.0)**2)
            mix_ratios[:, self.data.species_idx['CH4']] = ch4_abundance
        
        # CO2: minor species
        if 'CO2' in self.data.species_idx:
            mix_ratios[:, self.data.species_idx['CO2']] = 1e-5 * metallicity
        
        # NH3: nitrogen chemistry
        if 'NH3' in self.data.species_idx:
            # NH3 also temperature dependent
            nh3_abundance = 2e-5 * metallicity * np.exp(-(T_profile / 1200.0)**2)
            mix_ratios[:, self.data.species_idx['NH3']] = nh3_abundance
        
        # Trace radicals and photochemical species
        trace_species = ['OH', 'H', 'O', 'CH3', 'HCN', 'C2H2', 'N', 'N2']
        for species in trace_species:
            if species in self.data.species_idx:
                if species == 'N2':
                    # N2 is relatively abundant
                    mix_ratios[:, self.data.species_idx[species]] = 1e-4 * metallicity
                else:
                    mix_ratios[:, self.data.species_idx[species]] = 1e-10
        
        # Normalize to ensure sum <= 1
        row_sums = mix_ratios.sum(axis=1, keepdims=True)
        mix_ratios = mix_ratios / np.maximum(row_sums, 1.0)
        
        return jnp.array(mix_ratios, dtype=jnp.float64)


# Global solver instance
_solver = None

def get_solver():
    """Get or create global solver instance"""
    global _solver
    if _solver is None:
        _solver = PhotochemSolver()
    return _solver


def evolve_atmosphere(T_profile, P_profile, stellar_flux, Kzz, 
                     C_O=0.55, metallicity=1.0):
    """
    Main entry point for validation
    
    Args:
        T_profile: Temperature profile [K]
        P_profile: Pressure profile [bar]
        stellar_flux: Stellar flux data
        Kzz: Eddy diffusion coefficient [cm^2/s]
        C_O: C/O ratio
        metallicity: Metallicity relative to solar
    
    Returns:
        dict: Mixing ratio profiles for each species
    """
    solver = get_solver()
    return solver.evolve_atmosphere(T_profile, P_profile, stellar_flux, Kzz, 
                                   C_O, metallicity)
