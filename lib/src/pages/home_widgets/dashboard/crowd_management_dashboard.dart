import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

// Function to create Crowd Management Pie Chart and Crowd Management Line Bars for different platforms
Widget createCrowdManagementDashboard(var platform1CrowdManagementValue, Color platform1CrowdManagementColor, var platform1CrowdManagementRadius, var platform2CrowdManagementValue, Color platform2CrowdManagementColor, var platform2CrowdManagementRadius, var platform3CrowdManagementValue, Color platform3CrowdManagementColor, var platform3CrowdManagementRadius, var platform4CrowdManagementValue, Color platform4CrowdManagementColor, var platform4CrowdManagementRadius, var platform5CrowdManagementValue, Color platform5CrowdManagementColor, var platform5CrowdManagementRadius, var platform6CrowdManagementValue, Color platform6CrowdManagementColor, var platform6CrowdManagementRadius){
  return Column(
    children: [
      // Pie Chart
      Stack(
        children: [
          SizedBox(
            height: 250,
            child: PieChart(PieChartData(sections: [
              PieChartSectionData(
                  value: platform1CrowdManagementValue + 1,
                  color: platform1CrowdManagementColor,
                  showTitle: false,
                  radius: platform1CrowdManagementRadius),
              PieChartSectionData(
                  value: platform2CrowdManagementValue + 1,
                  color: platform2CrowdManagementColor,
                  showTitle: false,
                  radius: platform2CrowdManagementRadius),
              PieChartSectionData(
                  value: platform3CrowdManagementValue + 1,
                  color: platform3CrowdManagementColor,
                  showTitle: false,
                  radius: platform3CrowdManagementRadius),
              PieChartSectionData(
                  value: platform4CrowdManagementValue + 1,
                  color: platform4CrowdManagementColor,
                  showTitle: false,
                  radius: platform4CrowdManagementRadius),
              PieChartSectionData(
                  value: platform5CrowdManagementValue + 1,
                  color: platform5CrowdManagementColor,
                  showTitle: false,
                  radius: platform5CrowdManagementRadius),
              PieChartSectionData(
                  value: platform6CrowdManagementValue + 1,
                  color: platform6CrowdManagementColor,
                  showTitle: false,
                  radius: platform6CrowdManagementRadius),
            ])),
          ),

          Positioned.fill(
            child: Align(
                alignment: Alignment.center,
                child: Text("Crowd", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: primaryColor),)
            ),
          )
        ],
      ),

      const SizedBox(height: 50),

      // Crowd Management Bars
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
              // platform 1 Crowd Management percent bar
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
                            "${double.parse((platform1CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform1CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform1CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform1CrowdManagementColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 2 Crowd Management percent bar
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
                            "${double.parse((platform2CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform2CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform2CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform2CrowdManagementColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 3 Crowd Management percent bar
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
                            "${double.parse((platform3CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform3CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform3CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform3CrowdManagementColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 4 Crowd Management percent bar
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
                            "${double.parse((platform4CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform4CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform4CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform4CrowdManagementColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 5 Crowd Management percent bar
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
                            "${double.parse((platform5CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform5CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform5CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform5CrowdManagementColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 6 Crowd Management percent bar
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
                            "${double.parse((platform6CrowdManagementValue).toStringAsFixed(2))}%",
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
                                color: platform6CrowdManagementColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform6CrowdManagementValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform6CrowdManagementColor,
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
Widget createCrowdTypePieChart(var lessCrowdValue, Color lessCrowdColor, var normalCrowdValue, Color normalCrowdColor, var highCrowdValue, Color highCrowdColor, var veryHighCrowdValue, Color veryHighCrowdColor){
  return Column(
    children: [
      // Crowd Management Status Type Pie Chart (Row - 1)
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
                          value: lessCrowdValue + 1,
                          color: lessCrowdColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - lessCrowdValue + 1,
                          color: lessCrowdColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Less Crowd - ${double.parse((lessCrowdValue).toStringAsFixed(2))}%")
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
                          value: normalCrowdValue + 1,
                          color: normalCrowdColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - normalCrowdValue + 1,
                          color: normalCrowdColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Normal Crowd - ${double.parse((normalCrowdValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      // Crowd Management status Type Pie Chart (Row - 2)
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
                          value: highCrowdValue + 1,
                          color: highCrowdColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - highCrowdValue + 1,
                          color: highCrowdColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("High Crowd - ${double.parse((highCrowdValue).toStringAsFixed(2))}%")
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
                          value: veryHighCrowdValue + 1,
                          color: veryHighCrowdColor,
                          showTitle: false,
                          radius: 20),
                      PieChartSectionData(
                          value: 100 - veryHighCrowdValue + 1,
                          color: veryHighCrowdColor.withOpacity(0.25),
                          showTitle: false,
                          radius: 20),
                    ])),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
                Text("Very High - ${double.parse((veryHighCrowdValue).toStringAsFixed(2))}%")
              ],
            ),
          ],
        ),
      ),

      const SizedBox(height: 50),
    ],
  );
}
