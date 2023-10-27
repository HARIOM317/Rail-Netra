import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/pages/home_widgets/live_cctv.dart';
import 'package:railway/src/pages/home_widgets/platforms.dart';
import 'package:railway/src/utils/constants.dart';

class NotificationPage extends StatefulWidget {
  const NotificationPage({super.key});

  @override
  State<NotificationPage> createState() => _NotificationPageState();
}

class _NotificationPageState extends State<NotificationPage> {

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
        title: Text(
          "Notifications",
          style: TextStyle(
              color: primaryColor, fontWeight: FontWeight.bold, fontSize: 22),
        ),
      ),

      body: Column(
        children: [
          Expanded(
              child: StreamBuilder(
            stream: FirebaseFirestore.instance
                .collection('notifications')
                .orderBy('created_at', descending: true)
                .snapshots(),
            builder:
                (BuildContext context, AsyncSnapshot<QuerySnapshot> snapshot) {
              if (!snapshot.hasData || snapshot.data?.size == 0) {
                return Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Center(
                      child: Lottie.asset(
                          "assets/animations/no_notification.json",
                          animate: true,
                          width: 300),
                    ),
                    const Text(
                      "No Any New Notification!",
                      style: TextStyle(fontSize: 20, color: Colors.black54),
                    )
                  ],
                );
              }
              return ListView.builder(
                  physics: const BouncingScrollPhysics(),
                  itemCount: snapshot.data!.docs.length,
                  itemBuilder: (BuildContext context, int index) {
                    final data = snapshot.data!.docs[index];

                    return ListTile(
                      leading: data['title'] == "New Crime Detected"
                          ? Image.asset(
                              "assets/images/notification_icons/crime_detected.png",
                              width: 24,
                              height: 24,
                            )
                          : data['title'] == "CCTV Added"
                              ? Image.asset(
                                  "assets/images/notification_icons/cctv_added.png",
                                  width: 24,
                                  height: 24,
                                )
                              : data['title'] == "Trash Found"
                                  ? Image.asset(
                                      "assets/images/notification_icons/dustbin.png",
                                      width: 24,
                                      height: 24,
                                    )
                                  : data['title'] == "Missing Employee"
                                      ? Image.asset(
                                          "assets/images/notification_icons/employee_work.png",
                                          width: 24,
                                          height: 24,
                                        )
                                      : Image.asset(
                                          "assets/images/notification_icons/crowd.png",
                                          width: 24,
                                          height: 24,
                                        ),
                      title: Text("${data['title']} at ${data['time']}"),
                      subtitle: Text(data['body']),
                      trailing: Text(data['date']),
                      onTap: () {
                        if(data['title'] == "CCTV Added"){
                          goTo(context, const LiveCCTV());
                        } else {
                          goTo(context, const ChooseOption());
                        }
                      },
                    );
                  });
            },
          )),
        ],
      ),
    );
  }
}
