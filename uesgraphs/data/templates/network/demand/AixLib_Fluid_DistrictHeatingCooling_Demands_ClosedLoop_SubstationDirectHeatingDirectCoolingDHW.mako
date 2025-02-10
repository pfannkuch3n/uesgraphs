  AixLib.Fluid.DistrictHeatingCooling.Demands.ClosedLoop.SubstationDirectHeatingDirectCoolingDHW demand${str(name)}(
    redeclare package Medium = Medium,
    %if cp_default is not None:
    cp_default = ${str(round(cp_default, 4))},
    %endif
    %if dp_nominal is not None:
    dp_nominal = ${str(round(dp_nominal, 4))},
    %endif
    %if m_flow_nominal is not None:
    m_flow_nominal = ${str(round(m_flow_nominal, 4))},
    %endif
    heatDemand_max = ${str(round(heatDemand_max, 4))},
    deltaT_heatingSet = ${str(round(deltaT_heatingSet, 4))},
    deltaT_coolingGridSet = ${str(round(deltaT_coolingGridSet, 4))},
    T_supplyDHWSet = ${str(round(T_supplyDHWSet, 4))},
    T_returnSpaceHeatingSet = ${str(round(T_returnSpaceHeatingSet, 4))}
    )
    annotation(Placement(transformation(
      extent={{-2,-2},{2,2}},
      rotation=0,
      origin={${str(round(x, 4))},${str(round(y, 4))}})));

  Modelica.Blocks.Interfaces.RealInput ${str(name + 'heatDemand')}
    annotation(Placement(
      transformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${str(round(x+25, 4))},${str(round(y+25, 4))}}),
      iconTransformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${-100},${str(round(90 - i*180/(max(number_of_instances-1.0, 1.0)) + 0, 4))}})
      ));

  Modelica.Blocks.Interfaces.RealInput ${str(name + 'coolingDemand')}
    annotation(Placement(
      transformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${str(round(x+25, 4))},${str(round(y+15, 4))}}),
      iconTransformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${-100},${str(round(90 - i*180/(max(number_of_instances-1.0, 1.0)) + 10, 4))}})
      ));

  Modelica.Blocks.Interfaces.RealInput ${str(name + 'dhwDemand')}
    annotation(Placement(
      transformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${str(round(x+25, 4))},${str(round(y+5, 4))}}),
      iconTransformation(
        extent={{-2,-2},{2,2}},
        rotation=0,
        origin={${-100},${str(round(90 - i*180/(max(number_of_instances-1.0, 1.0)) + 20, 4))}})
      ));
<%def name="get_main_parameters()">
   heatDemand_max deltaT_heatingSet deltaT_coolingGridSet T_supplyDHWSet T_returnSpaceHeatingSet
</%def><%def name="get_aux_parameters()">
   cp_default dp_nominal m_flow_nominal
</%def><%def name="get_connector_names()">
   heatDemand coolingDemand dhwDemand
</%def>