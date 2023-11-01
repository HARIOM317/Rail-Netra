import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

// Function to create Work Monitoring Pie Chart and Work Monitoring Line Bars for different platforms
Widget createWorkMonitoringDashboard(var platform1WorkMonitoringValue, Color platform1WorkMonitoringColor, var platform1WorkMonitoringRadius, var platform2WorkMonitoringValue, Color platform2WorkMonitoringColor, var platform2WorkMonitoringRadius, var platform3WorkMonitoringValue, Color platform3WorkMonitoringColor, var platform3WorkMonitoringRadius, var platform4WorkMonitoringValue, Color platform4WorkMonitoringColor, var platform4WorkMonitoringRadius, var platform5WorkMonitoringValue, Color platform5WorkMonitoringColor, var platform5WorkMonitoringRadius, var platform6WorkMonitoringValue, Color platform6WorkMonitoringColor, var platform6WorkMonitoringRadius){
  return Column(
    children: [
      // Pie Chart
      Stack(
        children: [
          SizedBox(
            height: 250,
            child: PieChart(PieChartData(sections: [
              PieChartSectionData(
                  value: platform1WorkMonitoringValue + 1,
                  color: platform1WorkMonitoringColor,
                  showTitle: false,
                  radius: platform1WorkMonitoringRadius),
              PieChartSectionData(
                  value: platform2WorkMonitoringValue + 1,
                  color: platform2WorkMonitoringColor,
                  showTitle: false,
                  radius: platform2WorkMonitoringRadius),
              PieChartSectionData(
                  value: platform3WorkMonitoringValue + 1,
                  color: platform3WorkMonitoringColor,
                  showTitle: false,
                  radius: platform3WorkMonitoringRadius),
              PieChartSectionData(
                  value: platform4WorkMonitoringValue + 1,
                  color: platform4WorkMonitoringColor,
                  showTitle: false,
                  radius: platform4WorkMonitoringRadius),
              PieChartSectionData(
                  value: platform5WorkMonitoringValue + 1,
                  color: platform5WorkMonitoringColor,
                  showTitle: false,
                  radius: platform5WorkMonitoringRadius),
              PieChartSectionData(
                  value: platform6WorkMonitoringValue + 1,
                  color: platform6WorkMonitoringColor,
                  showTitle: false,
                  radius: platform6WorkMonitoringRadius),
            ])),
          ),

          Positioned.fill(
            child: Align(
                alignment: Alignment.center,
                child: Text("Truancy", style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: primaryColor),)
            ),
          )
        ],
      ),

      const SizedBox(height: 50),

      // Work Monitoring Bars
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
              // platform 1 Work Monitoring percent bar
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
                            "${double.parse((platform1WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform1WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform1WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform1WorkMonitoringColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 2 Work Monitoring percent bar
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
                            "${double.parse((platform2WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform2WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform2WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform2WorkMonitoringColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 3 Work Monitoring percent bar
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
                            "${double.parse((platform3WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform3WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform3WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform3WorkMonitoringColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 4 Work Monitoring percent bar
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
                            "${double.parse((platform4WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform4WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform4WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform4WorkMonitoringColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 5 Work Monitoring percent bar
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
                            "${double.parse((platform5WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform5WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform5WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform5WorkMonitoringColor,
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // platform 6 Work Monitoring percent bar
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
                            "${double.parse((platform6WorkMonitoringValue).toStringAsFixed(2))}%",
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
                                color: platform6WorkMonitoringColor.withOpacity(0.25),
                                borderRadius: BorderRadius.circular(10)),
                          ),
                          LayoutBuilder(
                            builder: (context, constraints) => Container(
                              width:
                              constraints.maxWidth * (platform6WorkMonitoringValue / 100),
                              height: 15,
                              decoration: BoxDecoration(
                                  color: platform6WorkMonitoringColor,
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