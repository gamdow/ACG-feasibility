# Landau–Lifshitz-Gilbert (LLG)

I'm solving the Landau–Lifshitz equation (which is equivalent to the LLG equation) and looks like;

$$\frac{\partial\vec{m}}{\partial t} = -\frac{\gamma^\star}{1+\alpha^2}\vec{m}\times\vec{H_{eff}(\vec{m})} - \frac{\alpha\gamma^\star}{1+\alpha^2}\vec{m}\times(\vec{m}\times\vec{H_{eff}(\vec{m})}) = \textrm{LLG}(\vec{m})$$

where $\vec{m} = \vec{m}(t,x,y,z)$ is the magnetisation, a vector field and $\vec{H_{eff}}(\vec{m})$ is the "effective field", an apparent vector field due to the interaction of the material and $\vec{m}$. The exact nature of $\vec{H_{eff}}$ depends on the material, but is generally a sum of the terms like the following;

| |Effective Field|Einstein Notation|
|-|:-:|:-:|
|Zeeman|$\vec{H}$   |$H_i$|
|Anisotropy|$\frac{2K}{\mu_0M_s}(\vec{m}.\vec{e})\vec{e}$|$\frac{2K}{\mu_0M_s}m_ke_ke_i$|
|Exchange|$\frac{2A}{\mu_0M_s}\nabla^2\vec{m}$|$\frac{2A}{\mu_0M_s}\partial_j\partial_jm_i$|
|DMI|$\frac{2D}{\mu_0M_s}\nabla\times\vec{m}$|$\frac{2D}{\mu_0M_s}\epsilon_{ijk}\partial_jm_k$|

# Runge Kutta

Then I'm applying the RK algorithm to solve it, so;
$$\textrm{temp}_1 = \vec{m}_n$$
$$k_1 = \textrm{LLG}(\textrm{temp}_1)$$
$$\textrm{temp}_2 = \vec{m}_n + s\frac{k_1}{2}$$
$$k_2 = \textrm{LLG}(\textrm{temp}_2)$$
$$\textrm{temp}_3 = \vec{m}_n + s(2k_2 - k_1)$$
$$k_3 = \textrm{LLG}(\textrm{temp}_3)$$
$$\vec{m}_{n+1} = \vec{m}_n + \frac{s}{6}(k_1 + 4k_2 + k_3)$$

My LLG code takes a single TimeFunction, so I've introduced three temporaries to calculate the three $(\vec{m}_n + \ldots)$ expressions passed to the LLG. I also used the standard solve method on $\frac{\partial\vec{m}}{\partial t} = \frac{1}{6}(k_1 + 4k_2 + k_3)$ to generate the last line without using explicit time indexing, but that was probably overkill.
