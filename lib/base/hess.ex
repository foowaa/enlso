alias :math, as: Math 
import Enlso.Base.Grad 
alias Enlso.Base.Grad, as: Grad
import Matrix 
defmodule Enlso.Base.Hess do
    @moduledoc """
    An implementation of auto Hessian matrix. See http://mathworld.wolfram.com/Hessian.html
    """
    defp finiteHess(f, x) do
        p = length(x)
        mu = 2 * Math.sqrt(1.0e-12) * (1+Grad.norm(x))
        diff = List.duplicate(0.0, p) |> List.duplicate(p)
        g = x|>Grad.grad!(f, "finite")
        for j <- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            e_j|>List.replace_at(j, 1.0)
            temp = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end) |> Grad.grad!(f,"finite")
            for i <- 0..p-1, do: diff |> Enum.at(i) |> List.replace_at(j, temp|>Enum.at(i))
        end
        rep_g = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: rep_g |> List.replace_at(i, g|>Enum.at(i)|>List.duplicate(p))
        rep_g = rep_g|>Matrix.transpose
        mu_list = List.duplicate(mu, p)
        h = diff |> Enum.zip(rep_g) |> Enum.map(fn(x) -> Tuple.to_list(x) end) |> 
                    Enum.map(fn(x) -> x|>Enum.at(0)|>Enum.zip(x|>Enum.at(1))|>Enum.map(fn(x)-> elem(x,0)-elem(x,1) end) end)
        hes1 = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: hes1|>List.replace_at(i, Enum.at(h,i)|>Enum.zip(mu_list)|>Enum.map(fn(x) -> elem(x,0)/elem(x,1) end))
        hes2 = hes1|>Matrix.add(Matrix.transpose(hes1))
        two = List.duplicate(0.50, p)
        hessian = List.duplicate(0, p) |> List.duplicate(p)
        for i<- 0..p-1, do: hessian|>List.replace_at(i, Enum.at(hes2, i)|>Enum.zip(two)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end))
    end

    defp centralHess(f, x) do   
        p = length(x)
        mu = 2 * Math.sqrt(1.0e-12) * (1+Grad.norm(x))
        diff1 = List.duplicate(0.0, p) |> List.duplicate(p)
        diff2 = List.duplicate(0.0, p) |> List.duplicate(p)
        
        for j<- 0..p-1 do
            e_j = List.duplicate(0.0, p)
            e_j|>List.replace_at(j, 1.0)
            g1 = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 0)+elem(x, 1) end) |> Grad.grad!(f)         
            g2 = mu |> List.duplicate(p) |> Enum.zip(e_j) |>
                Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end) |>
                Enum.zip(x) |> Enum.map(fn(x) -> elem(x, 1)-elem(x, 0) end) |> Grad.grad!(f)
            for i <- 0..p-1 do
                diff1 |> Enum.at(i) |> List.replace_at(j, g1|>Enum.at(i))
                diff2 |> Enum.at(i) |> List.replace_at(j, g2|>Enum.at(i))
            end 
        end
        mu_list = List.duplicate(2*mu, p)
        h = diff1 |> Enum.zip(diff2) |> Enum.map(fn(x) -> Tuple.to_list(x) end) |> 
                    Enum.map(fn(x) -> x|>Enum.at(0)|>Enum.zip(x|>Enum.at(1))|>Enum.map(fn(x)-> elem(x,0)-elem(x,1) end) end)
        hes1 = List.duplicate(0, p) |> List.duplicate(p)
        for i <- 0..p-1, do: hes1|>List.replace_at(i,Enum.at(h,i)|>Enum.zip(mu_list)|>Enum.map(fn(x) -> elem(x,0)/elem(x,1) end))
        hes2 = hes1|>Matrix.add(Matrix.transpose(hes1))
        two = List.duplicate(0.50, p)
        hessian = List.duplicate(0, p) |> List.duplicate(p)
        for i<- 0..p-1, do: hessian|>List.replace_at(i, Enum.at(hes2, i)|>Enum.zip(two)|>Enum.map(fn(x) -> elem(x, 0)*elem(x, 1) end))
    end
        @doc """
        There are 2 types of difference methods: finite AND central.
        finite differencing, see: http://mathworld.wolfram.com/FiniteDifference.html
        central differencing, see: http://mathworld.wolfram.com/CentralDifference.html

        ## Parameters:

        - x: list, the diff point
        - f: function handler
        - type: string which determines the method to calculate the diff, default is central.
        
        ## Examples

        iex> f = fn(x) x|>Enum.zip(x)|>Enum.map(fn(x)->elem(x,0)*elem(x,1) end)|>Enum.sum
        iex> Enlso.Base.Hess.hess!(f, [1,2,3], "finite")
        [[2.00,0.00,0.00],[0.00,2.00,0.00],[0.00,0.00,2.00]]
        iex> Enlso.Base.Hess.hess!(f, [1,2,3])
        [[2.00,0.00,0.00],[0.00,2.00,0.00],[0.00,0.00,2.00]]
        """  
    def hess!(x, f, type \\ "central") do
  
        cond do
            String.equivalent?(type, "central") ->
                hessian = centralHess(f, x)
            String.equivalent?(type, "finite") ->
                hessian = finiteHess(f, x)
            true ->
                raise ArgumentError, message: "type is invalid"
        end
    end
end