#!/usr/bin/env wolframscript
(* ::Package:: *)

X10=ToExpression[$ScriptCommandLine[[2]]];
X20=ToExpression[$ScriptCommandLine[[3]]];
S1=ToExpression[$ScriptCommandLine[[4]]];
S2=ToExpression[$ScriptCommandLine[[5]]];


signal={{S1}, {S2}}; (*signal={{1},{-1}}*)
dt=0.001;


W1=Transpose[{{1,2,3},{3,2,1}}];
W2=Transpose[{{1,0,-1},{1,1,2},{-2, 0, 1}}];
W3=Transpose[{{2},{3},{1}}];
WT= Transpose[{{2, -1},{4, 1}}];
act[x_]:=Tanh[x];


f1[x1_,x2_]:=act[Dot[W1,{{x1}, {x2}}]];
f2[y_]:=act[Dot[W2,y]];
f3[y_]:=Dot[W3,y];
Phi[x1_, x2_]:=f3[f2[f1[x1, x2]]][[1,1]];


X1=X10;
X2=X20;
(*Print["signal: ", signal]*)
(*Print["Phi: ",N[Phi[X1, X2]]]*)
dphidx=D[Phi[x1, x2], {x1, 1}]/.{x1->X1,x2->X2};
dphidy=D[Phi[x1, x2], {x2, 1}]/.{x1->X1,x2->X2};
(*Print["grad Phi: ", {N[dphidx], N[dphidy]}]*)
tilt = Dot[WT, signal];
(*Print["tilt Tau: ", tilt]*)
f=-({{dphidx}, {dphidy}} + tilt);
(*Print["F: ", N[f]]*)
dX=f*dt;
(*Print["dX: ", dX]*)
newX1=X1+dX[[1]];
newX2=X2+dX[[2]];
Print[Flatten[{newX1,newX2}]]


X1=newX1;
X2=newX2;
(*Print["signal: ", signal]*)
(*Print["Phi: ",N[Phi[X1, X2]]]*)
dphidx=D[Phi[x1, x2], {x1, 1}]/.{x1->X1,x2->X2};
dphidy=D[Phi[x1, x2], {x2, 1}]/.{x1->X1,x2->X2};
(*Print["grad Phi: ", {N[dphidx], N[dphidy]}]*)
tilt = Dot[WT, signal];
(*Print["tilt Tau: ", tilt]*)
f=-({{dphidx}, {dphidy}} + tilt);
(*Print["F: ", N[f]]*)
dX=f*dt;
(*Print["dX: ", dX]*)
newX1=X1+dX[[1]];
newX2=X2+dX[[2]];
(*Print[Flatten[{newX1,newX2}]]*)
