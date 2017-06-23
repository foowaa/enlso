alias :math, as Math 
import enlso.base.grad 
alias enlso.base.grad, as Grad
defmodule enlso.base.hess do
 @moduledoc """
 An implementation of auto hess
 """
    defp finiteHess(f, x) do
    @doc """
    finite differencing
    """
        p = length(x)
        mu = 2 * Math.sqrt(1.0e-12) * (1+Grad.norm(x))
        diff = List.duplicate(0.0, p) |> List.duplicate(p)

        for j <- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            Enum.at(e_j, j) = 1.0
            g = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end) |> Grad.grad(f,"finite")
            for i <- 0..p-1, do: diff |> Enum.at(i) |> List.replace_at(j, g|>Enum.at(i))
        end
        rep_g = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: rep_g |> List.replace_at(i, g|>Enum.at(i)|>List.duplicate(p))
        mu_list = List.duplicate(mu, p)
        h = diff |> Enum.zip(rep_g) |> Enum.map(fn(x) -> Tuple.to_list(x) end) |> 
                    Enum.map(fn(x) -> x|>Enum.at(0)|>Enum.zip(x|>Enum.at(1))|>Enum.map(fn(x)-> elem(x,0)-elem(x,1) end) end)
        hessian = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: Enum.at(hessian, i) = Enum.at(h,i)|>Enum.zip(mu_list)|>Enum.map(fn(x) -> elem(x,0)/elem(x,1) end)
    end

    defp centralHess(f, x) do
    @doc """
    central differencing
    """    
        p = length(x)
        mu = 2 * Math.sqrt(1.0e-12) * (1+Grad.norm(x))
        diff1 = List.duplicate(0.0, p) |> List.duplicate(p)
        diff2 = List.duplicate(0.0, p) |> List.duplicate(p)
        
        for j<- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            Enum.at(e_j, j) = 1.0
            g1 = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end) |> Grad.grad(f)         
            g2 = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 1)-elem(x, 0) end) |> Grad.grad(f)
        end
        mu_list = List.duplicate(2*mu, p)
        h = diff1 |> Enum.zip(diff2) |> Enum.map(fn(x) -> Tuple.to_list(x) end) |> 
                    Enum.map(fn(x) -> x|>Enum.at(0)|>Enum.zip(x|>Enum.at(1))|>Enum.map(fn(x)-> elem(x,0)-elem(x,1) end) end)
        hessian = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: hessian|>List.replace_at(i,Enum.at(h,i)|>Enum.zip(mu_list)|>Enum.map(fn(x) -> elem(x,0)/elem(x,1) end))
    end

    def hess!(x, f, type \\ "central") do
    @doc """
    the main part
    """    
        cond do
            String.equivalent?(type, "central") ->
                hessian = centralHess(f, x)
            String.equivalent?(type, "finite") ->
                hessian = finiteHess(f, x)
            _ ->
                raise ArgumentError, message:"type is invalid"
        end
    end
end