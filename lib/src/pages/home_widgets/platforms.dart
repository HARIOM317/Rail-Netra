import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/utils/constants.dart';
import 'package:railway/src/utils/live_stream.dart';

class ChooseOption extends StatefulWidget {
  const ChooseOption({super.key});

  @override
  State<ChooseOption> createState() => _ChooseOptionState();
}

class _ChooseOptionState extends State<ChooseOption> {
  static String? selectedPlatform;
  static String? selectedMonitoringType;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: primaryColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Image.asset(
            "assets/images/icons/logo.png",
            height: 200,
          ),
          Column(
            children: [
              platformDropdownList(
                  selectedPlatform: selectedPlatform,
                  callbackFunction: (String? newPlatform) {
                    setState(() {
                      selectedPlatform = newPlatform!;
                    });
                  }),
              monitoringTypeDropdownList(
                  selectedMonitoringType: selectedMonitoringType,
                  callbackFunction: (String? newMonitoringType) {
                    setState(() {
                      selectedMonitoringType = newMonitoringType!;
                    });
                  }),
            ],
          ),
          Container(
            margin: const EdgeInsets.all(20),
            height: 60,
            width: double.infinity,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                  backgroundColor: primaryColor,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30))),
              onPressed:
                  selectedPlatform != null && selectedMonitoringType != null
                      ? () {
                          goTo(context, const SelectedType());
                        }
                      : null,
              child: const Text(
                "Select",
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
              ),
            ),
          )
        ],
      ),
    );
  }
}

class SelectedType extends StatefulWidget {
  const SelectedType({super.key});

  @override
  State<SelectedType> createState() => _SelectedTypeState();
}

class _SelectedTypeState extends State<SelectedType> {
  @override
  Widget build(BuildContext context) {
    String? platform = _ChooseOptionState.selectedPlatform;
    String? monitoringType = _ChooseOptionState.selectedMonitoringType;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: primaryColor),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          "$platform - $monitoringType",
          style: TextStyle(
              color: primaryColor, fontWeight: FontWeight.bold, fontSize: 18),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: StreamBuilder(
                stream: monitoringType?.toLowerCase() == "crime"
                    ? FirebaseFirestore.instance
                        .collection('crime_videos')
                        .where('platform', isEqualTo: platform)
                        .snapshots()
                    : monitoringType?.toLowerCase() == "cleanliness"
                        ? FirebaseFirestore.instance
                            .collection('cleanliness_videos')
                            .where('platform', isEqualTo: platform)
                            .snapshots()
                        : monitoringType?.toLowerCase() == "work monitoring"
                            ? FirebaseFirestore.instance
                                .collection('work_monitoring_videos')
                                .where('platform', isEqualTo: platform)
                                .snapshots()
                            : monitoringType?.toLowerCase() == "crowd management"
                                ? FirebaseFirestore.instance
                                    .collection('crowd_management_videos')
                                    .where('platform', isEqualTo: platform)
                                    .snapshots()
                                : FirebaseFirestore.instance
                                    .collection('cctv_urls')
                                    .where('platform', isEqualTo: platform)
                                    .snapshots(),
                builder: (BuildContext context,
                    AsyncSnapshot<QuerySnapshot> snapshot) {
                  if (!snapshot.hasData || snapshot.data?.size == 0) {
                    return Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        const SizedBox(height: 100,),

                        Center(
                          child: Lottie.asset(
                              "assets/animations/no_data_found.json",
                              animate: true,
                              width: 300),
                        ),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 20.0),
                          child: Center(
                            child: Text(
                              "No CCTV Available on $platform",
                              style: TextStyle(
                                  fontSize: 20, color: primaryColor),
                            ),
                          ),
                        ),
                      ],
                    );
                  }
                  return ListView.builder(
                      physics: const BouncingScrollPhysics(),
                      itemCount: snapshot.data!.docs.length,
                      itemBuilder: (BuildContext context, int index) {
                        final data = snapshot.data!.docs[index];

                        return monitoringType?.toLowerCase() == "crime"
                            ? Column(
                                children: [
                                  Stack(
                                    children: [
                                      PlayVideoClip(url: data['video_url']),

                                      // Timestamp
                                      Positioned(
                                          top: 25,
                                          right: 25,
                                          child: Text("${data['date']} ${data['time']}",
                                            style: const TextStyle(
                                                color: Colors.white),
                                          ))
                                    ],
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Center(
                                      child: Text(
                                        "Crime Type: ${data['label']}",
                                        style: TextStyle(
                                            fontSize: 18,
                                            color: Colors.red[900]),
                                      ),
                                    ),
                                  ),
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Center(
                                      child: Text(
                                        "Accuracy: ${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                        style: TextStyle(
                                            fontSize: 18,
                                            color: Colors.green[900]),
                                      ),
                                    ),
                                  ),
                                  const SizedBox(
                                    height: 20,
                                  )
                                ],
                              )
                            : monitoringType?.toLowerCase() == "cleanliness"
                                ? Column(
                                    children: [
                                      Stack(
                                        children: [
                                          PlayVideoClip(url: data['video_url']),

                                          // Timestamp
                                          Positioned(
                                              top: 25,
                                              right: 25,
                                              child: Text(
                                                "${data['date']} ${data['time']}",
                                                style: const TextStyle(
                                                    color: Colors.white),
                                              ))
                                        ],
                                      ),
                                      Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: Center(
                                          child: Text(
                                            "Cleanliness Status: ${data['status']}",
                                            style: TextStyle(
                                                fontSize: 18,
                                                color: Colors.red[900]),
                                          ),
                                        ),
                                      ),
                                      Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: Center(
                                          child: Text(
                                            "Accuracy: ${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                            style: TextStyle(
                                                fontSize: 18,
                                                color: Colors.green[900]),
                                          ),
                                        ),
                                      ),
                                      const SizedBox(
                                        height: 20,
                                      )
                                    ],
                                  )
                                : monitoringType?.toLowerCase() == "work monitoring"
                                    ? Column(
                                        children: [
                                          Stack(
                                            children: [
                                              PlayVideoClip(
                                                  url: data['video_url']),

                                              // Timestamp
                                              Positioned(
                                                  top: 25,
                                                  right: 25,
                                                  child: Text(
                                                    "${data['date']} ${data['time']}",
                                                    style: const TextStyle(
                                                        color: Colors.white),
                                                  ))
                                            ],
                                          ),
                                          Padding(
                                            padding: const EdgeInsets.all(8.0),
                                            child: Center(
                                              child: Text(
                                                "Status : ${data['label']}",
                                                style: TextStyle(
                                                    fontSize: 18,
                                                    color: Colors.red[900]),
                                              ),
                                            ),
                                          ),
                                          Padding(
                                            padding: const EdgeInsets.all(8.0),
                                            child: Center(
                                              child: Text(
                                                "Accuracy: ${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                                style: TextStyle(
                                                    fontSize: 18,
                                                    color: Colors.green[900]),
                                              ),
                                            ),
                                          ),
                                          const SizedBox(
                                            height: 20,
                                          )
                                        ],
                                      )
                                    : monitoringType?.toLowerCase() == "crowd management"
                                        ? Column(
                                            children: [
                                              Stack(
                                                children: [
                                                  PlayVideoClip(
                                                      url: data['video_url']),

                                                  // Timestamp
                                                  Positioned(
                                                      top: 25,
                                                      right: 25,
                                                      child: Text(
                                                        "${data['date']} ${data['time']}",
                                                        style: const TextStyle(
                                                            color:
                                                                Colors.white),
                                                      ))
                                                ],
                                              ),
                                              Padding(
                                                padding:
                                                    const EdgeInsets.all(8.0),
                                                child: Center(
                                                  child: Text(
                                                    "Status : ${data['label']}",
                                                    style: TextStyle(
                                                        fontSize: 18,
                                                        color: Colors.red[900]),
                                                  ),
                                                ),
                                              ),
                                              Padding(
                                                padding:
                                                    const EdgeInsets.all(8.0),
                                                child: Center(
                                                  child: Text(
                                                    "Accuracy: ${double.parse((data['accuracy'] * 100).toStringAsFixed(2))}%",
                                                    style: TextStyle(
                                                        fontSize: 18,
                                                        color:
                                                            Colors.green[900]),
                                                  ),
                                                ),
                                              ),
                                              const SizedBox(
                                                height: 20,
                                              )
                                            ],
                                          )
                                        : LiveStream(
                                            url: data['urlLink'] + "/video");
                      });
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
