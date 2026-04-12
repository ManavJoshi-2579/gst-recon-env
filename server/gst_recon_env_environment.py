import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pydantic import ValidationError
from .models import Action, Observation, Invoice, GSTR2BEntry, ActionType

try:
    import numpy as np
except ImportError:
    np = None

class GSTReconEnv:
    BASE_DATE = datetime(2025, 1, 1)
    SEED = 42
    VALID_TASKS = {"easy", "medium", "hard"}

    def __init__(self, task: str = "easy"):
        self.task = task if task in self.VALID_TASKS else "easy"
        self.reset()

    def reset(self) -> Observation:
        random.seed(self.SEED)
        if np is not None:
            np.random.seed(self.SEED)
        self.episode_id = f"{self.task}-seed-{self.SEED}"
        self.step_count = 0
        self.last_reward = 0.0
        self.last_error = None
        self.matched = []
        self.mismatches = []
        self.claimed_itc = 0.0
        self.risk_score = 0.0
        self.wrong_itc_claims = 0
        self.final_score = 0.0
        self.last_info = {}
        self.processed_actions = set()
        self.processed_invoices = set()
        self.correct_decisions = set()
        self.action_history = []
        self.last_action_type = None
        self.same_action_streak = 0
        self.mismatch_prob = {"easy": 0.1, "medium": 0.3, "hard": 0.5}[self.task]
        self.missing_prob = {"easy": 0.02, "medium": 0.08, "hard": 0.12}[self.task]
        self.total_invoices = self._generate_num_invoices()
        self.invoices = self._generate_invoices()
        self.gstr2b = self._generate_gstr2b()
        self.current_idx = 0
        self.max_steps = self.total_invoices + 1
        self.warnings = []
        self.total_itc_possible = sum(
            inv.igst + inv.cgst + inv.sgst
            for inv in self.invoices
            if self._is_valid_invoice(inv)
        )
        
        obs = self._get_observation()
        obs.step_count = 0
        return obs

    def state(self) -> Dict:
        return {
            "invoices": [inv.model_dump() for inv in self.invoices],
            "processed": sorted(self.processed_invoices),
            "risk_score": self.risk_score,
            "steps": self.step_count,
        }

    def close(self):
        return None

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        reward = 0.0
        error = None
        done = False

        try:
            if self.step_count >= self.max_steps:
                reward = self._handle_submit_report() - 0.2
                done = True
            elif action.type == ActionType.SUBMIT_REPORT:
                reward = self._handle_submit_report()
                done = True
            else:
                idx = self.step_count % len(self.invoices)
                current_invoice = self.invoices[idx]
                action_key = (current_invoice.id, action.type.value)
                requested_key = (
                    (action.invoice_id, action.type.value)
                    if action.invoice_id and action.invoice_id != current_invoice.id
                    else None
                )
                valid = self._is_valid_invoice(current_invoice)

                if action.type == self.last_action_type:
                    self.same_action_streak += 1
                else:
                    self.same_action_streak = 1
                self.last_action_type = action.type

                if current_invoice.id in self.processed_invoices:
                    reward -= 0.3
                if action_key in self.processed_actions or (
                    requested_key is not None and requested_key in self.processed_actions
                ):
                    reward -= 0.1
                if self.same_action_streak >= 3:
                    reward -= 0.1

                if action_key not in self.processed_actions:
                    self.action_history.append(action.type)
                    self.processed_actions.add(action_key)
                    self.processed_invoices.add(current_invoice.id)

                    if action.type == ActionType.MATCH:
                        reward += 0.3 if valid else -0.3
                        if valid:
                            self.matched.append(current_invoice.id)
                            self.correct_decisions.add(current_invoice.id)
                        else:
                            self.mismatches.append(current_invoice.id)
                    elif action.type == ActionType.REJECT:
                        reward += 0.2 if not valid else -0.3
                        self.mismatches.append(current_invoice.id)
                        if not valid:
                            self.correct_decisions.add(current_invoice.id)
                    elif action.type == ActionType.CLAIM_ITC:
                        if valid:
                            self.claimed_itc += current_invoice.igst + current_invoice.cgst + current_invoice.sgst
                            self.matched.append(current_invoice.id)
                            self.correct_decisions.add(current_invoice.id)
                            reward += 0.5
                        else:
                            self.risk_score += 0.25
                            self.wrong_itc_claims += 1
                            self.warnings.append("INVALID ITC CLAIM")
                            reward -= 0.7
                    elif action.type == ActionType.QUERY_VENDOR:
                        reward += 0.05 if not valid else -0.1
                    else:
                        error = "Invalid action type"
                        reward -= 0.1

                self.step_count += 1
                self.current_idx = min(self.step_count, len(self.invoices))
                done = self.step_count >= self.total_invoices

        except ValidationError as e:
            error = f"Invalid action: {str(e)}"
            reward -= 0.2
        except Exception as e:
            error = str(e)
            reward -= 0.1

        self._update_warnings()
        self.last_reward = reward
        self.last_error = error
        score = self.compute_score() if done else 0.0
        score = min(max(score, 0.0), 1.0)
        self.final_score = round(score, 3)
        self.last_info = {
            "score": round(score, 3),
            "risk": self.risk_score,
            "processed": len(self.processed_invoices),
        }
        obs = self._get_observation()
        obs.step_count = self.step_count
        obs.warnings = self.warnings
        
        return obs, self.last_reward, done, self.last_info

    def _generate_num_invoices(self) -> int:
        if self.task == "easy": return 3
        if self.task == "medium": return 5
        return 8  # hard

    def _generate_invoices(self) -> List[Invoice]:
        invoices = []
        base_date = self.BASE_DATE
        
        for i in range(self.total_invoices):
            fraud_prob = {"easy": 0.02, "medium": 0.08, "hard": 0.15}[self.task]
            einvoice_prob = {"easy": 0.98, "medium": 0.92, "hard": 0.88}[self.task]
            is_fraud = random.random() < fraud_prob
            
            gstin = self._generate_gstin(is_fraud)
            hsn = f"{random.randint(1000,9999):04d}" if random.random() < 0.7 else f"{random.randint(100000,999999):06d}"
            value = round(random.uniform(1000, 50000), 2)
            igst_rate = random.choice([0, 5, 12, 18, 28]) / 100
            igst = round(value * igst_rate, 2)
            cgst = sgst = 0
            
            if igst == 0:
                rate = random.choice([0, 2.5, 6, 9, 14]) / 100
                cgst = sgst = round(value * rate, 2)
            
            inv = Invoice(
                id=f"INV-{i+1:03d}",
                gstin=gstin,
                date=(base_date + timedelta(days=i*2)).strftime("%Y-%m-%d"),
                value=value,
                hsn=hsn,
                igst=igst, cgst=cgst, sgst=sgst,
                is_einvoice=random.random() < einvoice_prob,
                is_fraud=is_fraud
            )
            invoices.append(inv)
        return invoices

    def _generate_gstr2b(self) -> List[GSTR2BEntry]:
        entries = []
        
        for inv in self.invoices:
            if random.random() < self.missing_prob:
                continue

            if random.random() > self.mismatch_prob:
                entries.append(GSTR2BEntry(
                    invoice_id=inv.id,
                    gstin=inv.gstin,
                    date=inv.date,
                    value=inv.value,
                    igst=inv.igst, cgst=inv.cgst, sgst=inv.sgst
                ))
            else:
                # Mismatch
                entries.append(GSTR2BEntry(
                    invoice_id=f"MISMATCH-{inv.id}",
                    gstin=self._generate_gstin(False),
                    date=(datetime.strptime(inv.date, "%Y-%m-%d") + timedelta(days=random.randint(-3,3))).strftime("%Y-%m-%d"),
                    value=round(inv.value * random.uniform(0.8, 1.2), 2),
                    igst=round(inv.igst * random.uniform(0.8, 1.2), 2),
                    cgst=round(inv.cgst * random.uniform(0.8, 1.2), 2),
                    sgst=round(inv.sgst * random.uniform(0.8, 1.2), 2)
                ))
        return entries

    def _generate_gstin(self, invalid: bool) -> str:
        if not invalid:
            state = random.randint(1, 37)
            pan = f"ABCDE{random.randint(1000,9999)}F"
            entity = chr(random.randint(65,90))
            checksum = "Z"  # simplified
            return f"{state:02d}{pan}{entity}{checksum}"
        else:
            return "INVALIDGSTIN123"  # obvious fraud

    def _get_observation(self) -> Observation:
        current_invoice = self.invoices[self.current_idx] if self.current_idx < len(self.invoices) else None
        progress = min(1.0, self.current_idx / len(self.invoices))
        return Observation(
            current_invoice=current_invoice,
            available_gstr2b=self.gstr2b,
            matched=self.matched,
            mismatches=self.mismatches,
            current_itc=self.claimed_itc,
            total_itc_possible=self.total_itc_possible,
            progress=progress
        )

    def _is_valid_invoice(self, inv: Invoice) -> bool:
        matching_gstr = next((g for g in self.gstr2b if g.invoice_id == inv.id), None)
        if not matching_gstr:
            return False
        tax_matches = (
            matching_gstr.igst == inv.igst
            and matching_gstr.cgst == inv.cgst
            and matching_gstr.sgst == inv.sgst
        )
        return (
            not inv.is_fraud
            and not inv.gstin.startswith("INVALID")
            and inv.is_einvoice
            and matching_gstr.gstin == inv.gstin
            and matching_gstr.date == inv.date
            and matching_gstr.value == inv.value
            and tax_matches
        )

    def _handle_match(self, action: Action) -> float:
        if not action.invoice_id:
            raise ValueError("invoice_id required for MATCH")
        
        inv = next((i for i in self.invoices if i.id == action.invoice_id), None)
        if not inv or inv.id in self.matched:
            return -0.3
        
        # Check if valid match exists in GSTR2B
        matching_gstr = next((g for g in self.gstr2b if g.invoice_id == inv.id), None)
        if matching_gstr:
            self.matched.append(inv.id)
            self.current_idx += 1
            return 0.25  # Good match
        else:
            self.mismatches.append(inv.id)
            return -0.15  # Wrong match

    def _handle_reject(self, action: Action) -> float:
        if not action.invoice_id:
            raise ValueError("invoice_id required for REJECT")
        
        inv = next((i for i in self.invoices if i.id == action.invoice_id), None)
        if not inv:
            return -0.1
        
        # Correct rejection gives partial credit
        has_match = any(g.invoice_id == inv.id for g in self.gstr2b)
        if not has_match or inv.is_fraud:
            self.mismatches.append(inv.id)
            self.current_idx += 1
            return 0.12
        return -0.08  # Wrong rejection

    def _handle_claim_itc(self, action: Action) -> float:
        if not action.invoice_id:
            raise ValueError("invoice_id required for CLAIM_ITC")
        
        inv = next((i for i in self.invoices if i.id == action.invoice_id), None)
        if not inv:
            return -0.4  # Heavy penalty for invalid claim
        
        tax = inv.igst + inv.cgst + inv.sgst
        has_valid_gstr = any(g.invoice_id == inv.id for g in self.gstr2b)
        
        if inv.is_fraud or inv.gstin.startswith("INVALID") or not inv.is_einvoice:
            self.warnings.append("FRAUD ITC CLAIM")
            return -0.40  # Heavy fraud penalty
        
        if has_valid_gstr:
            self.claimed_itc += tax
            return 0.25
        return -0.20  # Invalid claim

    def _handle_query_vendor(self, action: Action) -> float:
        if self.task == "hard":
            return 0.08  # Useful in hard task
        return 0.02  # Minor positive

    def _handle_submit_report(self) -> float:
        accuracy = len(self.correct_decisions) / max(1, self.total_invoices)
        return accuracy * 1.5 - min(self.risk_score, 1.0) * 0.5

    @property
    def correct_matches(self) -> int:
        return len(self.correct_decisions)

    def grade_easy(self) -> float:
        return self.correct_matches / max(1, len(self.invoices))

    def grade_medium(self) -> float:
        penalty = self.wrong_itc_claims * 0.2
        return max(0.0, (self.correct_matches / max(1, len(self.invoices))) - penalty)

    def grade_hard(self) -> float:
        risk_penalty = min(self.risk_score, 1.0)
        return max(0.0, (self.correct_matches / max(1, len(self.invoices))) * (1 - risk_penalty))

    def _task_grade(self) -> float:
        if self.task == "easy":
            return self.grade_easy()
        if self.task == "medium":
            return self.grade_medium()
        return self.grade_hard()

    def compute_score(self) -> float:
        if self.task == "easy":
            score = self.correct_matches / max(1, len(self.invoices))
        elif self.task == "medium":
            penalty = self.wrong_itc_claims * 0.2
            score = (self.correct_matches / max(1, len(self.invoices))) - penalty
        else:
            risk_penalty = min(self.risk_score, 1.0)
            score = (self.correct_matches / max(1, len(self.invoices))) * (1 - risk_penalty)
        return min(max(score, 0.0), 1.0)

    def _diversity_penalty(self) -> float:
        if not self.action_history:
            return 0.0

        dominant_action_count = max(self.action_history.count(action) for action in set(self.action_history))
        dominant_action_ratio = dominant_action_count / len(self.action_history)
        if dominant_action_ratio <= 0.55:
            return 0.0
        return (dominant_action_ratio - 0.55) * 1.2

    def _current_score_estimate(self) -> float:
        score = self._task_grade() - min(self.risk_score, 1.0) * 0.1 - self._diversity_penalty()
        return round(max(0.0, min(score, 1.0)), 2)

    def _update_warnings(self):
        self.warnings = []
        if self.claimed_itc > self.total_itc_possible * 1.05:
            self.warnings.append("ITC OVERCLAIM")
        if self.risk_score > 0:
            self.warnings.append(f"RISK SCORE {self.risk_score:.2f}")
        for inv in self.invoices:
            if inv.is_fraud and inv.id in self.matched:
                self.warnings.append("FRAUD MATCHED")

    def _calculate_grader_score(self) -> float:
        return round(self.compute_score(), 3)
