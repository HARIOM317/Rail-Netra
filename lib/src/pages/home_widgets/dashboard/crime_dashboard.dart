import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

// Function to create Crime Pie Chart and Crime Line Bars for different platforms
Widget createCrimeDashboard(var platform1Value, Color platform1Color, var platform1Radius, var platform2Value, Color platform2Color, var platform2Radius, var platform3Value, Color platform3Color, var platform3Radius, var platform4Value, Color platform4Color, var platform4Radius, var platform5Value, Color platform5Color, var platform5Radius, var platform6Value, Color platform6Color, var platform6Radius){
  return Column(
    children: [
      // Pie Chart
      Stack(
        children: [
          SizedBox(
            height: 250,
            child: PieChart(PieChartData(sections: [
              PieChartSectionData(
                  value: platform1Value + 1,
                  color: platform1Color,
                  showTitle: false,
                  radius: platform1Radius),
              PieChartSectionData(
                  value: platform2Value + 1,
                  color: platform2Color,
                  showTitle: false,
                  radius: platform2Radius),
              PieChartSectionData(
                  value: platform3Value + 1,
                  color: platform3Color,
                  showTitle: false,
                  radius: platform3Radius),
              PieChartSectionData(
                  value: platform4Value + 1,
                  color: platform4Color,
                  showTitle: false,
                  radius: platform4Radius),
              PieChartSectionData(
                  value: platform5Value + 1,
                  color: platform5Color,
                  showTitle: false,
                  radius: platform5Radius),
              PieChartSectionData(
                  value: platform6Value + 1,
                  color: platform6Color,
                  showTitle: false,
                  radius: platform6Radius),
            ])),
          ),

          Positioned.fill(
            child: Align(
                alignment: Alignment.center,
                child: Text("Crime", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: primaryColor),)
            ),
          )
        ],
      ),

      const SizedBox(height: 50),

      // Crime Bars
      Padding(
        padding: const EdgeInsets.all(15.0),
        child: Container(
          decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.black12),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.shade200,
                  blurRadius: 8,
                  spreadRadius: 1,
                )
              ]),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // platform 1 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 1",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform1Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform1Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform1Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform1Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 2 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 2",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform2Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform2Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform2Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform2Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 3 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 3",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform3Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform3Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform3Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform3Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 4 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 4",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform4Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform4Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform4Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform4Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 5 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 5",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform5Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform5Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform5Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform5Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 6 crime percent bar
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            "Platform 6",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "${double.parse((platform6Value).toStringAsFixed(2))}%",
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10.0, vertical: 8.0),
                      child: Stack(
                        children: [
                          Container(
                            width: double.infinity,
                            height: 15,
                            decoration: BoxDecoration(
                                color: platform6Color.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform6Value / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform6Color,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),

      const SizedBox(height: 10),
    ],
  );
}

// Function to create pie charts for different types of crime
Widget createCrimeTypePieChart(var arsonValue, Color arsonColor, var fightingValue, Color fightingColor, var shootingValue, Color shootingColor, var robberyValue, Color robberyColor, var normalValue, Color normalColor){
  return Column(
    children: [
      // Crime Type Pie Chart (Row - 1)
      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: arsonValue + 1,
                          color: arsonColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - arsonValue + 1,
                          color: arsonColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Arson - ${double.parse((arsonValue).toStringAsFixed(2))}%")
              ],
            ),

            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: fightingValue + 1,
                          color: fightingColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - fightingValue + 1,
                          color: fightingColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Fighting - ${double.parse((fightingValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      // Crime Type Pie Chart (Row - 2)
      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: shootingValue + 1,
                          color: shootingColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - shootingValue + 1,
                          color: shootingColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Shooting - ${double.parse((shootingValue).toStringAsFixed(2))}%")
              ],
            ),

            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: robberyValue + 1,
                          color: robberyColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - robberyValue + 1,
                          color: robberyColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Robbery - ${double.parse((robberyValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      // Crime Type Pie Chart (Row - 3)
      Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                Container(
                  height: 130,
                  width: 130,
                  decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.black12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.shade200,
                          blurRadius: 15,
                          spreadRadius: 3,
                        )
                      ]),
                  child: Padding(
                    padding: const EdgeInsets.all(10.0),
                    child: PieChart(PieChartData(sections: [
                      PieChartSectionData(
                          value: normalValue + 1,
                          color: normalColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - normalValue + 1,
                          color: normalColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Normal - ${double.parse((normalValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      const SizedBox(height: 50),
    ],
  );
}
