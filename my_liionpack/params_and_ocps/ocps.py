import numpy as np
import matplotlib.pyplot as plt
import os

def graphite_ocp_avg(sto):
    kB = 1.380649e-23  # J/K
    T = 298.15  # K
    e = 1.602176634e-19  # C
    e_o_kbT = e/(kB*T)
    
    width = 5e-2
    muLtail = -5e-2*1./(sto**(0.85))
    muRtail = 5e-2*1./((1.001-sto)**(0.85))
    muRtail = 1.0e1*0.5*(np.tanh((sto - 1.0)/0.045) + 1)

    muRlin = (0.90*1.6) * 0.5*(np.tanh((sto - 0.5)/(0.4*width)) + 1)

    muLMod = (0.
                + 40*(-np.exp(-sto/0.015))
                + 0.75*(np.tanh((sto-0.17)/0.02) - 1)
                + 1.0*(np.tanh((sto-0.22)/0.040) - 1)
                )* 0.5*(-np.tanh((sto - 0.35)/(width)) + 1)

    muR = 0.18 + muLMod + muLtail + muRtail + muRlin
    V = 0.12 - muR/e_o_kbT
    return V

def graphite_ocp_delithi(sto):
    kB = 1.380649e-23  # J/K
    T = 298.15  # K
    e = 1.602176634e-19  # C
    e_o_kbT = e/(kB*T)
    
    width = 5e-2
    muLtail = -5e-2*1./(sto**(0.85))
    muRtail = 5e-2*1./((1.001-sto)**(0.85))
    muRtail = 1.0e1*0.5*(np.tanh((sto - 1.0)/0.045) + 1)

    muRlin = (0.75*1.6) * 0.5*(np.tanh((sto - 0.5)/(0.4*width)) + 1) # ok
    
    muLMod = (0.
                + 40*(-np.exp(-sto/0.015))
                + 0.75*(np.tanh((sto-0.17)/0.02) - 1)
                + 1.0*(np.tanh((sto-0.22)/0.040) - 1)
                )* 0.5*(-np.tanh((sto - 0.35)/(width)) + 1)

    muR_bias = -0.3 * 0.5*(-np.tanh((sto - 0.5)/(0.4*width)) + 1) * 0.5*(np.tanh((sto - 0.25)/(0.4*width)) + 1)


    muR = 0.18 + muLMod + muLtail + muRtail + muRlin + muR_bias
    V = 0.12 - muR/e_o_kbT
    return V

def graphite_ocp_lithi(sto):
    kB = 1.380649e-23  # J/K
    T = 298.15  # K
    e = 1.602176634e-19  # C
    e_o_kbT = e/(kB*T)
    
    width = 5e-2
    muLtail = -5e-2*1./(sto**(0.85))
    muRtail = 5e-2*1./((1.001-sto)**(0.85))

    muRtail = 1.0e1*0.5*(np.tanh((sto - 1.0)/0.045) + 1)

    muRlin = (1.05*1.6) * 0.5*(np.tanh((sto - 0.5)/(0.4*width)) + 1) # ok
    
    muLMod = (0.
                + 40*(-np.exp(-sto/0.015))
                + 0.75*(np.tanh((sto-0.17)/0.02) - 1)
                + 1.0*(np.tanh((sto-0.22)/0.040) - 1)

                )* 0.5*(-np.tanh((sto - 0.35)/(width)) + 1)
    
    muR_bias = 0.15 * 0.5*(-np.tanh((sto - 0.5)/(0.4*width)) + 1) * 0.5*(np.tanh((sto - 0.25)/(0.4*width)) + 1)

    muR = 0.18 + muLMod + muLtail + muRtail + muRlin + muR_bias
    V = 0.12 - muR/e_o_kbT
    return V

def graphite_ocp_phase_field(sto):
    kB = 1.380649e-23  # J/K
    T = 298.15  # K
    e = 1.602176634e-19  # C
    e_o_kbT = e/(kB*T)

    Omga = 3.4
    Omgb = 1.6
    
    width = 5e-2
    muLtail = -5e-2*1./(sto**(0.85))
    muRtail = 5e-2*1./((1-sto)**(0.85))
    muRtail = 1.0e1*0.5*(np.tanh((sto - 1.0)/0.045) + 1)
    muLlin = (0.15*Omga*12*(0.40-sto**0.98)
                * 0.5*(-np.tanh((sto - 0.49)/(0.9*width)) + 1)
                * 0.5*(np.tanh((sto - 0.35)/width) + 1)) 
    muRlin = (0.1*Omga*4*(0.74-sto) + 0.90*Omgb)* 0.5*(np.tanh((sto - 0.5)/(0.4*width)) + 1)
    muLMod = (0.
                + 40*(-np.exp(-sto/0.015))
                + 0.75*(np.tanh((sto-0.17)/0.02) - 1)
                + 1.0*(np.tanh((sto-0.22)/0.040) - 1)
                )* 0.5*(-np.tanh((sto - 0.35)/(width)) + 1)

    muR = 0.18 + muLMod + muLtail + muRtail + muLlin + muRlin
    V = 0.12 - muR/e_o_kbT
    return V


def LFP_ocp_phase_field(sto):
    return 3.43 - 0.0257*(np.log(sto/(1-sto)) + 3.8*(1-2*sto))

def LFP_ocp_lithi(sto):
    c1 = -39.9676 * sto
    c2 = -500.0000 * (1 - sto)
    k = 3.4075 + (-0.0100 * sto) + (0.0668 * np.exp(c1)) + (-0.0787 * np.exp(c2))
    return k

def LFP_ocp_delithi(sto):
    c1 = -500.0000 * sto
    c2 = -39.9676 * (1 - sto)
    k = 3.4623 + (-0.0100 * sto) + (0.0787 * np.exp(c1)) + (-0.0668 * np.exp(c2))
    return k

def LFP_ocp_avg(sto):
    c1 = -150 * sto
    c2 = -150 * (1 - sto)
    k = (3.4075 + 3.4623)/2 - 0.01 * sto + (0.0787 + 0.0668)/2 * np.exp(c1) - (0.0787 + 0.0668)/2 * np.exp(c2)
    return k

if __name__ == "__main__":

    sto = np.linspace(0.001, 0.999, 1000)

    plt.figure()
    plt.plot(sto, graphite_ocp_avg(sto), label="Graphite OCP avg")
    plt.plot(sto, graphite_ocp_delithi(sto), label="Graphite OCP delithiation")
    plt.plot(sto, graphite_ocp_lithi(sto), label="Graphite OCP lithiation")

    plt.plot(sto, graphite_ocp_phase_field(sto),
            linestyle="--",color = 'black', label="Graphite OCP phase field")
    plt.ylim(0, 0.2)
    plt.xlim(0, 1)
    plt.xlabel("Stoichiometry")
    plt.ylabel("OCP [V]")
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "ocp_graphite.png"), dpi=300)

    plt.figure()
    plt.plot(sto, LFP_ocp_phase_field(sto), color = 'black',
            linestyle="--",
            label="LFP OCP phase field")


    plt.plot(sto, LFP_ocp_avg(sto), label="LFP OCP avg")
    plt.plot(sto, LFP_ocp_lithi(sto), label="LFP OCP lithiation")
    plt.plot(sto, LFP_ocp_delithi(sto), label="LFP OCP delithiation")
    plt.xlim(0, 1)
    plt.xlabel("Stoichiometry")
    plt.ylabel("OCP [V]")


    plt.legend()


    val = 0.005
    print(f"Value of LFP ocp phase field at {val}: " , LFP_ocp_phase_field(val))
    print(f"Value of LFP ocp avg at {val}: " , LFP_ocp_avg(val))
    print(f"Value of LFP ocp lithiation at {val}: " , LFP_ocp_lithi(val))
    print(f"Value of LFP ocp delithiation at {val}: " , LFP_ocp_delithi(val))
    # save in the directory of this script

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "ocp_lfp.png"), dpi=300)

    plt.show()