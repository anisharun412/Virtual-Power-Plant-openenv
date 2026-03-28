from typing import List

class VPPGrader:
    def __init__(self, transmission_fee_per_kwh: float = 0.01):
        """
        Initializes the grader for a single 96-step episode.
        """
        # OpenEnv Strict Safety Gates (Tracks over the whole episode)
        self.blackout_events_count = 0
        self.safety_violations_count = 0
        self.failed_hard_task_window = False
        
        # Financial Tracking
        self.transmission_fee_per_kwh = transmission_fee_per_kwh
        self.total_transmission_cost = 0.0
        self.total_raw_profit = 0.0

    def calculate_and_validate_step(
        self, 
        task_id: str, 
        action_charge_rate: float, 
        action_min_reserve: float, 
        batteries_soc: List[float], 
        grid_freq_hz: float, 
        market_price_mwh: float, 
        total_capacity_kwh: float
    ) -> float:
        """
        Dev 2 calls this every 15 minutes. 
        It tracks safety gates for the judges AND returns the RL training reward.
        """
        step_reward = 0.0

        # --- 1. TRACK OPENENV SAFETY GATES ---
        blackouts_this_step = sum(1 for soc in batteries_soc if soc <= 0.0)
        violations_this_step = sum(1 for soc in batteries_soc if soc < action_min_reserve)
        
        # Add to the episode running total
        self.blackout_events_count += blackouts_this_step
        self.safety_violations_count += violations_this_step

        # Hard Task Validation
        if task_id == "hard-frequency-response":
            if grid_freq_hz < 49.8 and action_charge_rate > -0.8:
                self.failed_hard_task_window = True

        # --- 2. CALCULATE FINANCIALS & TRANSMISSION FEES ---
        price_kwh = market_price_mwh / 1000.0 
        
        # Profit from arbitrage
        kwh_moved_raw = action_charge_rate * total_capacity_kwh * 0.25 
        financial_profit = -(kwh_moved_raw * price_kwh)
        self.total_raw_profit += financial_profit
        
        # Cost of moving the power (Efficiency/Transmission loss)
        kwh_absolute_moved = abs(action_charge_rate) * total_capacity_kwh * 0.25
        transmission_cost = kwh_absolute_moved * self.transmission_fee_per_kwh
        self.total_transmission_cost += transmission_cost
        
        # Agent's step reward is the profit minus the transmission fee
        step_reward += (financial_profit - transmission_cost)

        # --- 3. RL GRID SUPPORT & SAFETY PENALTIES ---
        if grid_freq_hz < 49.8:
            if action_charge_rate < 0:
                step_reward += 10.0  # Big bonus for saving the grid
            elif action_charge_rate > 0:
                step_reward -= 20.0  # Massive penalty for charging while grid dies

        if blackouts_this_step > 0:
            step_reward -= (50.0 * blackouts_this_step)
        elif violations_this_step > 0:
            step_reward -= (5.0 * violations_this_step)

        return step_reward

    def get_current_task_score(self, task_id: str, theoretical_max_profit: float) -> float:
        """
        Dev 1 calls this at Step 96 for the final deterministic OpenEnv score.
        """
        # Rule 2a & 3: The Strict Safety Gates
        if self.blackout_events_count > 0 or self.safety_violations_count > 0:
            return 0.0
        if task_id == "hard-frequency-response" and self.failed_hard_task_window:
            return 0.0

        # Rule 2b: Profit Scaling (Net Profit = Raw Profit - Transmission Costs)
        if theoretical_max_profit <= 0:
            return 0.0 
            
        actual_net_profit = self.total_raw_profit - self.total_transmission_cost
        
        # Ensure they don't get a negative final score if fees ate all their profits
        actual_net_profit = max(0.0, actual_net_profit) 
        
        score = actual_net_profit / theoretical_max_profit
        
        return max(0.0, min(1.0, float(score)))