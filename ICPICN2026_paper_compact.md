---
title: Safety-Projected Federated Graph Learning for EV Charging Coordination
author:
  - Lewei Yang, University of Science and Technology of China, School of Science and Technology, Department of Automation, ylw_yang@mail.ustc.edu.cn
date: ""
---

**Abstract**--

Electric-vehicle charging coordination is a cyber-physical edge-intelligence problem: charging stations must satisfy user deadlines, avoid distribution-network voltage violations, preserve local session privacy, and operate under limited communication. This paper proposes Grid-FedGAT-Safe, a safety-projected federated graph-learning method for active-distribution-network charging control. The feeder and stations are represented as a graph, a station-level graph-feature policy is trained by federated sufficient statistics, and every raw learned command is passed through a convex safety projection with deadline slack and a scalar AC-feasibility filter. The theory proves feasibility of the deployed command under a nonempty hard safety set, gives an AC-feasible fallback under a feasible no-EV base case, and shows exact recovery of the centralized ridge solution for the linear graph-feature pilot without sharing raw charging sessions. Validation spans a 15-scenario synthetic feeder sweep, nonlinear radial AC replay, MATPOWER case33bw/case69 replay, IEEE 123-bus OpenDSS replay, public Palo Alto ChargePoint session replay, and reserve sensitivity analysis. The selected controller achieves zero proposed-method voltage violations in all reported AC validations, wins delivered-energy service against graph FedAvg in all synthetic scenarios and all repeated public-session window scenarios, and reduces estimated training communication by 90.8% relative to full-precision graph FedAvg. The evidence supports a Pareto claim: improved safety, service, and communication efficiency, while charging cost remains scenario-dependent.

Keywords--EV charging coordination; active distribution networks; federated learning; graph learning; safety projection; communication-efficient edge intelligence.

# Introduction

Large-scale EV charging increases the magnitude and volatility of distribution-level demand [1], [2]. Uncontrolled charging can satisfy users but may concentrate load during stressed feeder periods; centralized optimization can mitigate this but often requires raw station sessions, local load states, and operator-specific records [3]-[5]. Federated learning is attractive because each station can train locally and share compact model information rather than raw sessions. However, a learned policy by itself is not enough for power-system deployment: it must be made safe with respect to voltage, transformer, charger, and deadline constraints.

Grid-FedGAT-Safe is designed as a safety-first learning architecture. A federated graph-feature policy predicts a preferred station-level charging action, while a convex projection and an AC-feasibility screen determine the command that is actually deployed. This separation allows graph-learning or reinforcement-learning proposals to be incorporated without relying on unconstrained learned outputs for physical feasibility.

The contributions are: (1) a safety-projected federated graph-learning framework for privacy-preserving EV charging coordination; (2) feasibility guarantees for convex grid projection, service slack, and scalar AC fallback; (3) exact one-shot federated sufficient-statistic learning for the linear graph-feature pilot; (4) communication accounting against multi-round FedAvg; and (5) reproducible synthetic, public-feeder, and public-session validation with reserve sensitivity.

# Related Work and Gap

Federated learning has been studied for mobile and edge systems because communication is often the limiting resource rather than centralized compute. FedAvg and FedProx reduce communication or improve heterogeneity tolerance [9], [10], but their standard objectives do not enforce feeder voltages, charger limits, or user-deadline feasibility at deployment time. Smart-grid federated-learning surveys and EV-control studies identify privacy, non-IID data, and adversarial robustness as important barriers [6]-[8], [11], yet many controller papers still evaluate learning accuracy or operating cost without an explicit post-learning safety certificate.

Graph learning is also a natural fit for power networks because the feeder topology and electrical distance between charging stations affect voltage sensitivity. GCN and GAT models provide standard graph-learning backbones [12], [13], and power-system GNN work for optimal power flow and state estimation shows that topology-aware models can generalize better than unstructured predictors [14], [15]. The limitation for this paper's setting is that a topology-aware predictor is still only a predictor: nonconvex training, distribution shift, and forecast errors can produce commands that are physically invalid unless a deployment layer checks them.

Unlike unconstrained graph-learning controllers, the proposed approach integrates communication-auditable federated graph-feature learning with a convex projection and nonlinear AC screen. The separation allows service gains to be evaluated through the learned proposal and reserve design, while safety is enforced by an explicit projection/filter layer with stated assumptions.

# Model and Method

Consider $K$ charging stations over slots $t=0,\ldots,T-1$. Let $p_t \in \mathbb{R}_+^K$ be station charging power and $\bar p_t$ the station power limits. Around a forecast base operating point, a linearized distribution model gives

$$v_t = v_t^0 - H_t p_t,$$

where $v_t^0$ is the base voltage and $H_t$ is the voltage-sensitivity matrix, motivated by linearized radial distribution-flow approximations [16]. The hard safety set is

$$\mathcal{C}_t = \{p: 0 \le p \le \bar p_t,\ H_t p \le v_t^0 - \underline v - \delta_v\mathbf{1}\}.$$

The additional feeder constraints are $A_t p \le b_t$ and $\mathbf{1}^\top p \le \bar P_t$.

Here $\delta_v$ is a deployment reserve against mismatch between the linear screen and nonlinear AC replay. For an active EV session $j$, let $e_{j,t}$ be remaining energy, $d_j$ the departure slot, $\bar r_j$ the maximum rate, and $\Delta t$ the slot duration. The one-step charging floor needed to preserve future deadline feasibility at station $k$ is

$$\ell_{k,t}=\sum_{j\in\mathcal{S}_{k,t}} \left[\frac{e_{j,t}-\bar r_j\Delta t(d_j-t-1)}{\Delta t}\right]_+.$$

Grid congestion can make every deadline impossible, so the controller uses nonnegative slack $s_t$. The service-aware set is $\mathcal{D}_t=\{(p,s):p\in\mathcal{C}_t,\ p+s\ge\ell_t,\ s\ge0\}$.

Let $G$ be the station graph and $X_t$ station features containing demand, urgency, tariff, base-load state, time features, and graph-neighbor summaries. A learned graph policy outputs $a_t=\pi_w(G,X_t)$. The proposal used by the projection adds a tariff-tiered discretionary reserve:

$$\hat a_{k,t}=\max\{a_{k,t},\ell_{k,t}+\eta_{\max}\omega(c_t)\bar p_{k,t}\},$$

with $\eta_{\max}=0.60$ and $\omega(c_t)=1.00$ for $c_t\le0.14$, $0.50$ for $0.14<c_t\le0.18$, and $0.05$ for $c_t>0.18$. The deployed first-stage command solves

$$ (p_t^\star,s_t^\star)=\arg\min_{(p,s)\in\mathcal{D}_t} \frac{1}{2}\|p-\hat a_t\|_W^2 + \rho\mathbf{1}^\top s,$$

where $W\succ0$ and $\rho>0$. For nonlinear AC validation, a scalar filter returns $\tilde p_t=\alpha_t^\star p_t^\star$, where

$$\alpha_t^\star=\max\{\alpha\in[0,1]:\Phi_t(\alpha p_t^\star)\ge\underline v\}.$$

The scalar search is implemented by bisection and falls back to zero EV charging if no positive scaling is feasible.

The deployed slot routine is therefore simple enough for edge implementation. At the start of slot $t$, each station forms local graph features and computes its raw proposal. The controller then computes the deadline floor $\ell_t$, lifts the proposal by the tariff-tiered reserve, solves the convex projection over $\mathcal{D}_t$, and runs the scalar AC screen. Only the screened command $\tilde p_t$ is sent to chargers. After the slot, remaining energy and departure states are updated locally, while training records remain at the station. This ordering is important: learning affects the preferred point in the projection objective, but grid feasibility is enforced after learning and before actuation.

For the pilot, $\pi_w$ is linear in graph features. Client $k$ uploads sufficient statistics $S_k=X_k^\top X_k$ and $h_k=X_k^\top y_k$, and the server solves the global ridge problem from $\sum_k S_k$ and $\sum_k h_k$. Raw sessions are not uploaded. With feature dimension $d$, $K$ clients, and 32-bit values, one-shot communication is

$$B_{\mathrm{stat}}=4K\left(\frac{d(d+1)}{2}+d\right)+4Kd.$$

A full-precision FedAvg baseline with $R$ bidirectional rounds requires $B_{\mathrm{FedAvg}}=8RKd$. In the pilot setting $d=17$ and $R=60$, the sufficient-statistic ratio is 0.0917, a 90.8% reduction.

This aggregation reduces communication by transmitting low-dimensional sufficient statistics rather than full neural parameter streams. It also separates statistical learning from safety enforcement: the federated estimator provides a preferred operating point, and the projection/filter layer enforces grid and service constraints before actuation.

# Feasibility Guarantees

Assume $W\succ0$, $\mathcal{C}_t$ is nonempty, and the linear grid constraints are used with calibrated reserve $\delta_v$. Then $\mathcal{D}_t$ is nonempty because any $p\in\mathcal{C}_t$ can be paired with $s\ge[\ell_t-p]_+$. The objective is continuous, convex, and coercive in $p$, while the positive linear slack penalty prevents unbounded slack from being optimal. Therefore an optimizer exists; strict convexity in $p$ implies the projected charging action $p_t^\star$ is unique. Since $(p_t^\star,s_t^\star)\in\mathcal{D}_t$, every hard grid constraint in $\mathcal{C}_t$ is satisfied by construction.

This gives the main deployment theorem. For any raw proposal $a_t$ produced by a learned, rule-based, or adversarially poor policy, if $\mathcal{C}_t$ is nonempty and the projection is solved to tolerance, the applied pre-AC command satisfies charger bounds, transformer or line limits, and the reserved linear voltage floor. The theorem deliberately says nothing about optimality of $a_t$; the learned model only chooses a point that the safety layer may pull back to feasibility. This is the central reason the method is provable even though graph learning itself is not globally optimizable.

The slack variable gives a service interpretation. If $s_t^\star=0$, then $p_t^\star\ge\ell_t$ and the one-step deadline-preserving lower bound is met. If no point in $\mathcal{C}_t$ can satisfy $p\ge\ell_t$, slack quantifies the service relaxation chosen under the penalty $\rho$, rather than hiding infeasibility.

For nonlinear AC deployment, assume the no-EV base case is AC-feasible and that, along the nonnegative EV scaling ray, increasing $\alpha$ does not increase the minimum monitored voltage. Then the feasible set of scaling factors is an interval containing zero. Bisection therefore returns a feasible $\alpha_t^\star$ up to tolerance; if $\alpha=1$ is feasible, the AC filter leaves the projected command unchanged. These results prove feasibility of the deployed command under explicit assumptions, not global optimality of a learned policy.

For learning, the sufficient-statistic aggregation exactly reconstructs the centralized ridge normal equations for the linear graph-feature pilot. Thus, aside from numerical solver tolerance, the one-shot federated estimator equals the centralized estimator that would have been obtained by pooling features and targets. Nonlinear GNN or RL extensions require standard nonconvex federated assumptions and are not used for the proof of deployment feasibility.

The privacy claim is also limited and auditable. The station uploads second-order feature statistics and cross-products, not raw arrival times, departure times, energy requests, or individual charging trajectories. These statistics can still leak information under a strong inference attacker if feature spaces are small or if a station has very few sessions, so the claim is communication-efficient federated training rather than formal differential privacy. A production deployment can add secure aggregation or noise, but those extensions are outside the current proof.

Two consequences follow. First, the proof does not require the raw graph policy to be stable, monotone, or calibrated; it only requires the projection problem to be solvable and the AC filter to be monotone along the tested ray. Second, the proof survives the replacement of the pilot learner by another proposal generator, because the projection and the fallback filter are the parts that enforce physics. That is the reason the paper can keep the learning model flexible while keeping the deployed action mathematically controlled.

# Experimental Design

The validation uses identical test scenarios across policies. Baselines are uncontrolled charging, earliest-deadline-first safe charging, FedAvg with local features, FedAvg with graph features, a no-guard ablation, and the proposed Grid-FedGAT-Safe controller. Metrics are delivered-energy service rate, unmet energy, charging cost, peak load, voltage violations, runtime, and estimated training communication.

Experiments include: (1) a 15-scenario synthetic sweep with three feeder-stress profiles and five random seeds; (2) nonlinear radial AC replay of synthetic schedules; (3) IEEE 33-bus, MATPOWER case33bw/case69, and IEEE 123-bus OpenDSS validation using standard power-system tools and test-feeder references [17]-[19]; (4) public Palo Alto ChargePoint session replay over a first held-out window and five separated train/test windows spanning 2011-2020 [20]; and (5) reserve sensitivity comparing hard-floor-only, cost-lean, selected, and flat-high reserve settings.

The baselines isolate distinct performance factors. Uncontrolled charging exposes the raw feasibility and safety tradeoff. Earliest-deadline-first safe charging evaluates a nonlearning service-oriented rule. FedAvg with local features and FedAvg with graph features provide communication-aware learning references. The no-guard ablation tests the effect of removing the convex projection. Together these baselines separate topology information, federated learning, and safety enforcement.

# Results

Table I summarizes the main service and safety evidence. The proposed controller eliminates voltage violations in the reported safety validations and improves service against graph FedAvg in all synthetic and repeated public-session window scenarios.

**Table I. Main service and safety evidence.**

| Validation | Key proposed result |
|---|---|
| Robust synthetic, 15 scenarios | Service 84.50 +/- 13.54%, cost 760.50 +/- 99.69, peak 197.75 +/- 20.59, violations 0.00%, projection runtime 1.678 +/- 0.422 ms |
| Synthetic service gates | Service wins vs graph FedAvg in 100% of scenarios; proposed is cheaper than EDF in 100%; mean cost vs graph FedAvg is about -0.8% to -1.0% depending on aggregation |
| Public Palo Alto first window | Nominal service 100.00%, unmet 0.00 kWh, cost 218.97, violations 0.00%; 4x stress service 93.90%, unmet 71.76 kWh, violations 0.00% |
| Repeated public windows | Nominal service 99.26 +/- 1.18%, stressed service 74.68 +/- 19.34%, zero violations, service gain 6.38 pp vs graph FedAvg and 2.17 pp vs no-guard |
| Communication | One-shot sufficient statistics use 9.17% of full graph FedAvg training traffic, a 90.8% reduction |

Table II summarizes AC replay. The proposed method has zero voltage violations after reserve projection and AC filtering.

**Table II. AC replay and feasibility-filter evidence.**

| AC validation | Proposed result |
|---|---|
| Synthetic radial AC | Min voltage 0.9629 +/- 0.0104; violations 0.000 +/- 0.000% |
| IEEE 33-bus replay | Min voltage 0.9519 +/- 0.0004; violations 0.000 +/- 0.000% |
| MATPOWER case33bw | Min voltage 0.9502 +/- 0.0002; violations 0.000 +/- 0.000%; retained 99.98 +/- 0.04%; runtime 0.114 +/- 0.006 ms |
| MATPOWER case69 | Min voltage 0.9516 +/- 0.0001; violations 0.000 +/- 0.000%; retained 100.00 +/- 0.00%; runtime 0.201 +/- 0.002 ms |
| IEEE 123-bus OpenDSS | Min voltage 0.9759 +/- 0.0003; violations 0.000 +/- 0.000%; retained 100.00%; runtime 0.055 +/- 0.001 ms |

Table III gives the reserve ablation that prevents the selected tariff tiers from looking arbitrary. The selected reserve is the smallest tested setting that satisfies the robust all-scenario service-win gate while retaining zero violations.

**Table III. Deadline-reserve sensitivity.**

| Reserve case | Observed tradeoff |
|---|---|
| Robust synthetic, cost-lean max 0.40, tiers 1.00/0.25/0.00 | 0.66 pp service gain and 93.3% service wins vs graph FedAvg; -0.55% cost; 100% zero-violation scenarios |
| Robust synthetic, selected max 0.60, tiers 1.00/0.50/0.05 | 1.78 pp service gain and 100% service wins vs graph FedAvg; -0.80% cost; 100% zero-violation scenarios |
| Robust synthetic, flat-high max 0.60, tiers 1.00/1.00/1.00 | 4.33 pp service gain and 100% service wins vs graph FedAvg; +12.75% cost; 100% zero-violation scenarios |
| Public windows, selected max 0.60, tiers 1.00/0.50/0.05 | 6.38 pp service gain and 100% service wins vs graph FedAvg; +5.63% cost; 100% zero violations |
| Public windows, flat-high max 0.60, tiers 1.00/1.00/1.00 | 7.88 pp service gain and 100% service wins vs graph FedAvg; +12.15% cost; 100% zero violations |

Reserve sensitivity explains the cost-service tradeoff. In robust synthetic replay, the selected reserve gives a 1.78 pp service gain versus graph FedAvg, 100% service wins, -0.80% cost delta, and 100% zero-violation scenarios. In repeated public windows, it gives a 6.38 pp service gain and 100% service wins, but cost is 5.63% higher than graph FedAvg. A weaker cost-lean reserve reduces some cost pressure but misses the all-scenario robust service-win gate; a flat high reserve improves service further only with about a 12% cost premium. The empirical evidence therefore supports a service-safety-communication tradeoff rather than universal charging-cost dominance.

The runtime results indicate that the projection and AC filter are computationally lightweight in the tested settings. This supports their use as online safety layers, although deployment on production station controllers would require additional software hardening and field validation.

The repeated public-window replay complements the synthetic sweep by testing whether service improvement persists under changes in time window and demand composition. The consistency of the repeated-window results provides evidence that the selected reserve is not tuned only to a single held-out period.

# Discussion and Limitations

The safety layer is the central contribution. Learned actions are never applied directly, so feasibility does not rely on perfect generalization by a graph model. The theory also separates what is proved from what is empirical: projection feasibility and sufficient-statistic exactness are proved under stated assumptions, while service and cost improvements are evaluated experimentally.

A limitation of the method is that it does not guarantee the lowest charging cost across all scenarios. The selected reserve improves service reliability and preserves safety, but public-window charging cost is higher on average than graph FedAvg. The appropriate interpretation is therefore a service-safety-communication tradeoff rather than cost dominance.

The validation design uses separate replay paths for separate empirical questions. The robust synthetic sweep tests controlled stress variation; radial AC, IEEE 33-bus, MATPOWER 33/69-bus, and IEEE 123-bus runs test whether the safety layer survives nonlinear power-flow checks; the Palo Alto replay tests whether service behavior persists on public charging sessions; and the reserve-sensitivity study tests the effect of tariff-tiered reserve choices. These experiments do not prove universal optimality or differential privacy. They support the narrower conclusion that a safety-first, communication-efficient controller can produce consistent service gains while maintaining grid feasibility under the tested conditions.

# Conclusion

This paper presented Grid-FedGAT-Safe, a federated graph-feature charging controller with a convex safety projection and scalar AC feasibility guard. The derivation proves existence, uniqueness, grid safety, service-slack interpretation, AC-feasible fallback, and exact one-shot federated ridge aggregation for the linear pilot. Synthetic, public-feeder, and public-session validations show zero proposed-method voltage violations in the reported safety checks, consistent service gains, and large communication savings. Future work will extend the proposal model beyond the linear pilot, evaluate larger public charging datasets, and study privacy mechanisms such as secure aggregation or differential privacy.

# References

[1] International Energy Agency, Global EV Outlook 2026, 2026. https://www.iea.org/reports/global-ev-outlook-2026

[2] J. A. Pecas Lopes, F. J. Soares, and P. M. R. Almeida, Integration of Electric Vehicles in the Electric Power System, Proceedings of the IEEE, vol. 99, no. 1, pp. 168-183, 2011. doi: 10.1109/JPROC.2010.2066250.

[3] L. Gan, U. Topcu, and S. H. Low, Optimal Decentralized Protocol for Electric Vehicle Charging, IEEE Transactions on Power Systems, vol. 28, no. 2, pp. 940-951, 2013. doi: 10.1109/TPWRS.2012.2210288.

[4] Z. Ma, D. S. Callaway, and I. A. Hiskens, Decentralized Charging Control of Large Populations of Plug-in Electric Vehicles, IEEE Transactions on Control Systems Technology, vol. 21, no. 1, pp. 67-78, 2013. doi: 10.1109/TCST.2011.2174059.

[5] J. Wang, H. Zhong, Z. Ma, Q. Xia, and C. Kang, Coordinated Electric Vehicle Charging With Reactive Power Support to Distribution Grids, IEEE Transactions on Industrial Informatics, vol. 15, no. 1, pp. 54-63, 2019. doi: 10.1109/TII.2018.2829710.

[6] L. Yan, X. Chen, J. Zhou, Y. Chen, and J. Wen, Deep Reinforcement Learning for Continuous Electric Vehicles Charging Control With Dynamic User Behaviors, IEEE Transactions on Smart Grid, vol. 12, no. 6, pp. 5124-5134, 2021. doi: 10.1109/TSG.2021.3096198.

[7] J. Qian, Y. Jiang, X. Liu, Q. Wang, T. Wang, Y. Shi, and W. Chen, Federated Reinforcement Learning for Electric Vehicles Charging Control on Distribution Networks, IEEE Internet of Things Journal, vol. 11, no. 3, pp. 5511-5525, 2024. doi: 10.1109/JIOT.2023.3306826.

[8] B. Feng, H. Xu, G. Huang, Z. Liu, C. Guo, and Z. Chen, Byzantine-Resilient Economical Operation Strategy Based on Federated Deep Reinforcement Learning for Multiple Electric Vehicle Charging Stations Considering Data Privacy, Journal of Modern Power Systems and Clean Energy, vol. 12, no. 6, pp. 1957-1967, 2024. doi: 10.35833/MPCE.2023.000850.

[9] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, Communication-Efficient Learning of Deep Networks from Decentralized Data, in Proc. AISTATS, PMLR, vol. 54, pp. 1273-1282, 2017.

[10] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, Federated Optimization in Heterogeneous Networks, in Proc. MLSys, 2020.

[11] Z. Zhang, S. Rath, J. Xu, and T. Xiao, Federated Learning for Smart Grid: A Survey on Applications and Potential Vulnerabilities, ACM Transactions on Cyber-Physical Systems, vol. 10, no. 1, pp. 1-26, 2026. doi: 10.1145/3760788.

[12] T. N. Kipf and M. Welling, Semi-Supervised Classification with Graph Convolutional Networks, in Proc. ICLR, 2017.

[13] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, Graph Attention Networks, in Proc. ICLR, 2018.

[14] W. Liao, B. Bak-Jensen, J. R. Pillai, Y. Wang, and Y. Wang, A Review of Graph Neural Networks and Their Applications in Power Systems, Journal of Modern Power Systems and Clean Energy, vol. 10, no. 2, pp. 345-360, 2022. doi: 10.35833/MPCE.2021.000058.

[15] D. Owerko, F. Gama, and A. Ribeiro, Optimal Power Flow Using Graph Neural Networks, in Proc. IEEE ICASSP, pp. 5930-5934, 2020. doi: 10.1109/ICASSP40776.2020.9053140.

[16] L. Gan and S. H. Low, Convex Relaxations and Linear Approximation for Optimal Power Flow in Multiphase Radial Networks, in Proc. Power Systems Computation Conference, pp. 1-9, 2014. doi: 10.1109/PSCC.2014.7038399.

[17] R. D. Zimmerman, C. E. Murillo-Sanchez, and R. J. Thomas, MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education, IEEE Transactions on Power Systems, vol. 26, no. 1, pp. 12-19, 2011. doi: 10.1109/TPWRS.2010.2051168.

[18] W. H. Kersting, Radial Distribution Test Feeders, in Proc. IEEE Power Engineering Society Winter Meeting, pp. 908-912, 2001. doi: 10.1109/PESW.2001.916993.

[19] R. C. Dugan and T. E. McDermott, An Open Source Platform for Collaborating on Smart Grid Research, in Proc. IEEE PES General Meeting, pp. 1-7, 2011. doi: 10.1109/PES.2011.6039829.

[20] Z. J. Lee, T. Li, and S. H. Low, ACN-Data: Analysis and Applications of an Open EV Charging Dataset, in Proc. ACM e-Energy, pp. 139-149, 2019. doi: 10.1145/3307772.3328313.
