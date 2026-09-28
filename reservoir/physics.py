"""Conservative finite-volume three-phase teaching simulator.

Incompressible, immiscible oil/water/gas. No gravity, capillarity, PVT,
dissolution, residual saturation or relative-permeability calibration.
Pressure is solved using total mobility; explicit upwind phase transport
uses adaptive positivity-preserving substeps, without saturation clipping.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

CONVERSION = 9.869233e-16 * 1e5 * 86400 / 1e-3  # mD, bar, cP -> m3/day


@dataclass(frozen=True)
class Scenario:
    rate: float = 15.0
    gas_fraction: float = 0.0
    days: float = 180.0
    initial_gas: float = 0.1
    pressure_bar: float = 200.0

    def validate(self):
        values = [self.rate, self.gas_fraction, self.days, self.initial_gas, self.pressure_bar]
        if not np.isfinite(values).all():
            raise ValueError("All scenario inputs must be finite")
        if not (0 <= self.rate <= 100 and 0 <= self.gas_fraction <= 1 and 0 < self.days <= 720
                and 0 <= self.initial_gas <= 0.7 and self.pressure_bar > 0):
            raise ValueError("Scenario inputs outside supported bounds")


class FlowModel:
    def __init__(self, rock):
        self.shape = rock["porosity"].shape
        self.n = int(np.prod(self.shape))
        spacing = np.asarray(rock["spacing"], float)
        if len(self.shape) != 3 or min(self.shape) < 2 or np.any(spacing <= 0):
            raise ValueError("Need a positive 3D grid with at least two cells on each axis")
        phi = rock["porosity"].ravel()
        if not np.isfinite(phi).all() or np.any((phi <= 0) | (phi > 1)):
            raise ValueError("Simulation porosity must be in (0, 1]")
        self.pv = phi * np.prod(spacing)
        indices = np.arange(self.n).reshape(self.shape)
        left, right, trans = [], [], []
        for axis, key in enumerate(("permz", "permy", "permx")):
            k = np.asarray(rock[key], float)
            if k.shape != self.shape or not np.isfinite(k).all() or np.any(k <= 0):
                raise ValueError("Permeability must be positive and finite")
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            harmonic = 2 * k[lo] * k[hi] / (k[lo] + k[hi])
            left.extend(indices[lo].ravel())
            right.extend(indices[hi].ravel())
            trans.extend((CONVERSION * harmonic * np.prod(spacing) / spacing[axis] ** 2).ravel())
        self.i, self.j, self.base_t = np.asarray(left), np.asarray(right), np.asarray(trans)
        self.injectors, self.producers = indices[:, 0, 0], indices[:, -1, -1]

    def pressure(self, s, rate, reference=200.0):
        # Phase order water, oil, gas. Assumed viscosities in cP.
        mobility = s ** 2 / np.array([1.0, 3.0, 0.08])
        total = mobility.sum(axis=1)
        fractional = mobility / total[:, None]
        t = self.base_t * 2 * total[self.i] * total[self.j] / (total[self.i] + total[self.j])
        i, j = self.i, self.j
        matrix = coo_matrix((np.r_[t, t, -t, -t], (np.r_[i, j, i, j], np.r_[i, j, j, i])),
                            shape=(self.n, self.n)).tocsr()
        source = np.zeros(self.n)
        source[self.injectors] = rate / len(self.injectors)
        source[self.producers] = -rate / len(self.producers)
        p = np.zeros(self.n)
        p[:-1] = spsolve(matrix[:-1, :-1], source[:-1])
        flux = t * (p[i] - p[j])
        residual = np.max(np.abs(matrix @ p - source))
        if not np.isfinite(p).all() or residual > 1e-7 * max(rate, 1):
            raise RuntimeError("Pressure solve failed conservation tolerance")
        return p + reference, flux, fractional, source, residual

    def run(self, scenario=Scenario(), snapshots=31):
        scenario.validate()
        if snapshots < 2:
            raise ValueError("At least two snapshots required")
        s = np.tile([0.2, 0.8 - scenario.initial_gas, scenario.initial_gas], (self.n, 1))
        injected_mix = np.array([1 - scenario.gas_fraction, 0, scenario.gas_fraction])
        initial = (s * self.pv[:, None]).sum(axis=0)
        cumulative = np.zeros(3)
        elapsed, steps, max_residual = 0.0, 0, 0.0
        states, pressures, rows = [], [], []
        for target in np.linspace(0, scenario.days, snapshots):
            while elapsed < target - 1e-10:
                p, flux, f, source, residual = self.pressure(s, scenario.rate, scenario.pressure_bar)
                max_residual = max(max_residual, residual)
                upstream = np.where(flux >= 0, self.i, self.j)
                downstream = np.where(flux >= 0, self.j, self.i)
                phase_flux = np.abs(flux[:, None]) * f[upstream]
                outgoing, incoming = np.zeros_like(s), np.zeros_like(s)
                np.add.at(outgoing, upstream, phase_flux)
                np.add.at(incoming, downstream, phase_flux)
                produced = -source[self.producers, None] * f[self.producers]
                outgoing[self.producers] += produced
                incoming[self.injectors] += source[self.injectors, None] * injected_mix
                capacity = s * self.pv[:, None]
                ratios = np.divide(capacity, outgoing, out=np.full_like(s, np.inf), where=outgoing > 1e-14)
                dt = min(target - elapsed, 0.75 * float(ratios.min()))
                if dt < 1e-10 or steps > 30000:
                    raise RuntimeError("Transport timestep too small; reduce duration/rate")
                s += dt * (incoming - outgoing) / self.pv[:, None]
                if np.min(s) < -1e-9 or np.max(np.abs(s.sum(axis=1) - 1)) > 1e-7:
                    raise RuntimeError("Saturation bounds or phase closure failed")
                cumulative += dt * produced.sum(axis=0)
                elapsed += dt
                steps += 1
            p, _, f, source, _ = self.pressure(s, scenario.rate, scenario.pressure_bar)
            rates = (-source[self.producers, None] * f[self.producers]).sum(axis=0)
            inventory = (s * self.pv[:, None]).sum(axis=0)
            balance = inventory - initial - scenario.rate * target * injected_mix + cumulative
            rows.append({"day": target, "water_m3_day": rates[0], "oil_m3_day": rates[1],
                         "gas_m3_day": rates[2], "water_cut": rates[0] / scenario.rate if scenario.rate else 0,
                         "oil_recovery": cumulative[1] / initial[1], "cumulative_oil_m3": cumulative[1],
                         "balance_error_m3": float(np.max(np.abs(balance))), "max_pressure_bar": p.max()})
            states.append(s.reshape((*self.shape, 3)).copy())
            pressures.append(p.reshape(self.shape))
        return {"history": pd.DataFrame(rows), "saturation": np.asarray(states),
                "pressure": np.asarray(pressures), "steps": steps, "pressure_residual": max_residual,
                "pore_volume_m3": self.pv.sum(), "initial_oil_m3": initial[1]}
