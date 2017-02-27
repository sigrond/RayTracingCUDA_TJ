#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
int main(int argc, char ** argv)
{
	int sock;
	struct sockaddr_un server;
	sock = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sock < 0) {
		perror("opening stream socket");
		exit(1);
	}
	server.sun_family = AF_UNIX;
	strcpy(server.sun_path, argv[1]);


	if (connect(sock, (struct sockaddr *) &server, sizeof(struct sockaddr_un)) < 0) {
		close(sock);
		perror("Error while connecting to a stream socket, did you start a server?");
		exit(1);
	}
	int dummy =1;
	if ( write(sock, &dummy, 1*sizeof(int)) < 0 )
		perror("Error writing to a stream socket");
	if (read(sock, &dummy, 1*sizeof(int)) < 0 )
		perror("Error reading stream stocket");
	close(sock);
	return 0;
}
