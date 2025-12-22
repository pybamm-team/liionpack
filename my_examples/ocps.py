from build_battery import *


# def graphite_LGM50_ocp_Chen2020(sto):
#     """
#     LG M50 Graphite open-circuit potential as a function of stoichiometry, fit taken
#     from [1].

#     References
#     ----------
#     .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
#     Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
#     Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
#     Electrochemical Society 167 (2020): 080534.

#     Parameters
#     ----------
#     sto: :class:`pybamm.Symbol`
#         Electrode stoichiometry

#     Returns
#     -------
#     :class:`pybamm.Symbol`
#         Open-circuit potential
#     """

#     u_eq = (
#         1.9793 * np.exp(-39.3631 * sto)
#         + 0.2482
#         - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
#         - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
#         - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
#     )

#     return u_eq


# def graphite_ocp_flat(sto):
#     kB = 1.380649e-23  # J/K
#     T = 298.15  # K
#     e = 1.602176634e-19  # C
#     e_o_kbT = e/(kB*T)

#     Omga = 0
#     Omgb = 1.6
    
#     width = 5e-2
#     muLtail = -5e-2*1./(sto**(0.85))
#     muRtail = 5e-2*1./((1-sto)**(0.85))
#     muRtail = 1.0e1*0.5*(np.tanh((sto - 1.0)/0.045) + 1)
#     muLlin = (0.15*Omga*12*(0.40-sto**0.98)
#                 * 0.5*(-np.tanh((sto - 0.49)/(0.9*width)) + 1)
#                 * 0.5*(np.tanh((sto - 0.35)/width) + 1)) 
#     muRlin = (0.1*Omga*4*(0.74-sto) + 0.90*Omgb)* 0.5*(np.tanh((sto - 0.5)/(0.4*width)) + 1)
#     muLMod = (0.
#                 + 40*(-np.exp(-sto/0.015))
#                 + 0.75*(np.tanh((sto-0.17)/0.02) - 1)
#                 + 1.0*(np.tanh((sto-0.22)/0.040) - 1)
#                 )* 0.5*(-np.tanh((sto - 0.35)/(width)) + 1)

#     muR = 0.18 + muLMod + muLtail + muRtail + muLlin + muRlin
#     V = 0.12 - muR/e_o_kbT
#     return V

# sto = np.linspace(0, 1, 100)
# ocp_graph = graphite_ocp_flat(sto)

# param_battery = pybamm.ParameterValues("Prada2013")

# chen_ocp = graphite_LGM50_ocp_Chen2020(sto)


# plt.plot(sto, ocp_graph)
# plt.plot(sto/1.2, chen_ocp, linestyle="--")



def LFP_ocp(sto):
    return 3.43 - 0.0257*(np.log(sto/(1-sto)) + 3.8*(1-2*sto))


def LFP_ocp_Afshar2017(sto):

    c1 = -150 * sto
    c2 = -30 * (1.1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)

    return k

sto = np.linspace(0.01, 0.99, 100)
lfp = LFP_ocp(sto)
lfp_afshar = LFP_ocp_Afshar2017(sto)
plt.plot(sto, lfp, label="LFP OCP Ombrini2024")
plt.plot(sto, LFP_ocp_Afshar2017(sto), linestyle="--", label="LFP OCP Afshar2017")
plt.legend()

plt.show()